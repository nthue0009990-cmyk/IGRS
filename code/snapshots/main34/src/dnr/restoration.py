"""D030 finite-horizon restoration bookkeeping and deterministic search.

No solver imports. Positive constant loads, one source, permanent edge outages.
The 32-state relaxation retains every healthy original edge, permits cycles and
ignores electrical limits. Mapping a real path to its ever-closed ties proves
the relaxed supplied set contains the real supplied set at every step.
"""
from functools import lru_cache
import heapq


def graph_state(branches, opened, faults=(), n=69):
    opened=frozenset(opened); faults=frozenset(faults)
    if not faults <= opened: raise ValueError('Faulted edge re-energized')
    if not opened <= {int(r['branch_id']) for r in branches}: raise ValueError('Unknown edge')
    parent=list(range(n+1)); graph={i:[] for i in range(1,n+1)}; forest=True
    def root(a):
        while parent[a]!=a: parent[a]=parent[parent[a]];a=parent[a]
        return a
    for r in branches:
        k=int(r['branch_id']);a=int(r['from_bus']);b=int(r['to_bus'])
        if k in opened:continue
        aa,bb=root(a),root(b)
        if aa==bb: forest=False
        else:parent[aa]=bb
        graph[a].append(b);graph[b].append(a)
    seen={1};todo=[1]
    while todo:
        for b in graph[todo.pop()]:
            if b not in seen:seen.add(b);todo.append(b)
    return dict(forest=forest,energized=sorted(seen),opened=sorted(opened))


def unsupplied_kw(graph, p_mw):
    live=set(graph['energized'])
    return sum(float(p)*1000 for i,p in enumerate(p_mw,1) if i not in live)


def neighbors(branches, opened, faults, p_mw, visited=()):
    """All healthy one-switch forest moves; exact connectivity-based ordering."""
    opened=frozenset(opened); prior=set(visited); result=[]
    for r in branches:
        k=int(r['branch_id'])
        if k in faults:continue
        nxt=tuple(sorted(opened ^ {k}))
        if nxt in prior:continue
        g=graph_state(branches,nxt,faults)
        if g['forest']:result.append((unsupplied_kw(g,p_mw),k,nxt))
    return sorted(result)


def bill(states, state_values, horizon):
    """States include initial, actions finish at interval end; holds pad to H."""
    assert 1<=len(states)<=horizon+1
    u=[state_values[tuple(s)]['unsupplied_kw'] for s in states]
    j=sum(u[:-1])+(horizon-(len(states)-1))*u[-1]
    return dict(j_kw_step=j,terminal_unsupplied_kw=u[-1],actions=len(states)-1,
                states=[list(s) for s in states],unsupplied_kw=u)


def relaxed_bound(branches, faults, p_mw, horizon=8, ties=(69,70,71,72,73)):
    """Exact DP for a relaxation, NOT an AC-feasible reference trajectory."""
    ties=tuple(ties);cost={}
    for mask in range(1<<len(ties)):
        opened=set(faults)|{e for j,e in enumerate(ties) if not mask>>j&1}
        cost[mask]=unsupplied_kw(graph_state(branches,opened,faults),p_mw)
    @lru_cache(None)
    def dp(mask,left):
        if left==0:return (0.,cost[mask],())
        options=[]
        for action,newmask in [(0,mask)]+[(e,mask|1<<j) for j,e in enumerate(ties) if not mask>>j&1]:
            tail=dp(newmask,left-1)
            options.append((cost[mask]+tail[0],tail[1],(action,)+tail[2]))
        return min(options)
    answer=dp(0,horizon)
    return dict(lower_bound_kw_step=answer[0],relaxed_lexicographic_terminal_unsupplied_kw=answer[1],
                actions=list(answer[2]),state_count=len(cost),ignores_cycles_voltage_and_current=True)


def search(branches,faults,p_mw,evaluate,method,budget=32,horizon=8,beam_width=4,
           checkpoints=(16,24,32),preferred=()):
    """Own observation cache per method. Evaluator may have a separately billed
    physical cache, but this method never reads evaluations it did not request.
    Every feasible prefix provides a complete hold-padded fallback for free.
    Repeated states along one path are dominated for fixed loads and this cost.
    """
    if method not in ('greedy','beam','best_first','reference'):raise ValueError(method)
    if budget<1 or horizon<1:raise ValueError('Positive budget and horizon required')
    initial=tuple(sorted(set(range(69,74))|set(faults))); own={};trace=[];snaps={};best=None
    def request(state):
        state=tuple(state)
        if state not in own:
            if len(own)>=budget:return None
            own[state]=evaluate(state)
            trace.append(dict(ordinal=len(own),opened=list(state),feasible=own[state]['feasible']))
        return own[state]
    def consider(path):
        nonlocal best
        cand=bill(path,own,horizon)
        if best is None:best=cand
        elif (cand['j_kw_step']<best['j_kw_step']-1e-6 and
              cand['terminal_unsupplied_kw']<=best['terminal_unsupplied_kw']+1e-8):best=cand
        elif (abs(cand['j_kw_step']-best['j_kw_step'])<=1e-6 and
              cand['terminal_unsupplied_kw']<best['terminal_unsupplied_kw']-1e-8):best=cand
        if len(own) in checkpoints:snaps[len(own)]=dict(best)
    def mark():
        if len(own) in checkpoints and len(own) not in snaps:snaps[len(own)]=dict(best)
    first=request(initial)
    if not first['feasible']:return dict(status='INITIAL_BLOCKED',trace=trace,checkpoints={},best=None)
    consider((initial,))
    # A separate reference arm first attempts the exact topological relaxation.
    if method=='reference':
        path=(initial,)
        for a in preferred:
            if not a:continue
            nxt=tuple(sorted(set(path[-1])^{a}))
            if not graph_state(branches,nxt,faults)['forest']:break
            r=request(nxt)
            if r is None or not r['feasible']:mark();break
            path=path+(nxt,);consider(path)
    if method=='greedy':
        path=(initial,)
        while len(path)<=horizon and len(own)<budget:
            advanced=False
            for u,a,nxt in neighbors(branches,path[-1],faults,p_mw,path):
                if u>=own[path[-1]]['unsupplied_kw']-1e-8:break
                r=request(nxt)
                if r is None:break
                if r['feasible']:
                    path=path+(nxt,);consider(path);advanced=True;break
                mark()
            if not advanced:break
    elif method=='beam':
        layer=[(initial,)]
        for depth in range(horizon):
            proposals=[]
            for path in layer:
                g=sum(own[x]['unsupplied_kw'] for x in path)
                for u,a,nxt in neighbors(branches,path[-1],faults,p_mw,path):
                    proposals.append((g+(horizon-depth-1)*u,u,a,path,nxt))
            next_layer=[];layer_states=set()
            for _,u,a,path,nxt in sorted(proposals):
                if nxt in layer_states:continue
                layer_states.add(nxt)
                r=request(nxt)
                if r is None:break
                if r['feasible']:
                    newpath=path+(nxt,);consider(newpath);next_layer.append(newpath)
                else:mark()
                if len(next_layer)>=beam_width:break
            layer=next_layer
            if not layer or len(own)>=budget:break
    else:
        # Best-first over untested path extensions, not just greedy descent.
        queue=[];counter=0;expanded={}
        def expand(path):
            nonlocal counter
            if len(path)>horizon:return
            g=sum(own[x]['unsupplied_kw'] for x in path);key=(path[-1],len(path))
            if key in expanded and expanded[key]<=g+1e-10:return
            expanded[key]=g;remaining=horizon-len(path)
            for u,a,nxt in neighbors(branches,path[-1],faults,p_mw,path):
                counter+=1;heapq.heappush(queue,(g+remaining*u,u,a,counter,path,nxt))
        expand((initial,))
        while queue and len(own)<budget:
            _,u,a,_,path,nxt=heapq.heappop(queue);r=request(nxt)
            if r is None:break
            if r['feasible']:
                newpath=path+(nxt,);consider(newpath);expand(newpath)
            else:mark()
    for b in checkpoints:
        if b not in snaps:snaps[b]=dict(best)
    return dict(status='COMPLETE_BOUNDED',method=method,evaluations=len(own),trace=trace,
                checkpoints={str(k):v for k,v in snaps.items()},best=best)
