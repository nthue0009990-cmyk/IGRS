"""Exact, deterministic depth-two candidate generation and linear voltage ranking."""
from fractions import Fraction as F
from dnr.case69_witness import cycle_neighbors, exhaustive_neighbors


def profile(opens, branches, loads, base_kv=F('12.66')):
    nodes={int(r['from_bus']) for r in branches}|{int(r['to_bus']) for r in branches}
    adj={i:[] for i in nodes}
    for r in branches:
        if int(r['branch_id']) not in opens:
            a,b=int(r['from_bus']),int(r['to_bus']);z=(F(r['r_ohm']),F(r['x_ohm']))
            adj[a].append((b,z));adj[b].append((a,z))
    parent={1:None};order=[1];imp={}
    for a in order:
        for b,z in adj[a]:
            if b==parent[a]:continue
            assert b not in parent
            parent[b]=a;imp[b]=z;order.append(b)
    assert len(order)==len(nodes)
    p={i:F(0) for i in nodes};q=p.copy()
    for r in loads:p[int(r['bus'])]+=F(r['p_kw'])/1000;q[int(r['bus'])]+=F(r['q_kvar'])/1000
    for b in reversed(order[1:]):p[parent[b]]+=p[b];q[parent[b]]+=q[b]
    u={1:F(1)}
    for b in order[1:]:
        rr,xx=imp[b];u[b]=u[parent[b]]-2*(rr*p[b]+xx*q[b])/base_kv**2
    return u


def integer_score(opens, branches, loads):
    # Independent representation: integer ohms * 10^4 and kW/kvar * 100.
    adj={i:[] for i in range(1,70)}
    def integer(value,scale):
        v=F(value)*scale;assert v.denominator==1;return v.numerator
    for r in branches:
        if int(r['branch_id']) not in opens:
            a,b=int(r['from_bus']),int(r['to_bus']);z=(integer(r['r_ohm'],10000),integer(r['x_ohm'],10000))
            adj[a].append((b,z));adj[b].append((a,z))
    # Recursive subtree aggregation, unlike the forward/reverse traversal above.
    subtree={};parent={};imp={}
    demand={i:[0,0] for i in adj}
    for r in loads:demand[int(r['bus'])]=[integer(r['p_kw'],100),integer(r['q_kvar'],100)]
    def visit(a,previous):
        assert a not in parent;parent[a]=previous;p,q=demand[a]
        for b,z in adj[a]:
            if b!=previous:
                imp[b]=z;bp,bq=visit(b,a);p+=bp;q+=bq
        subtree[a]=(p,q);return p,q
    visit(1,None);assert len(parent)==69
    costs={}
    for a in adj:
        cost=0;b=a
        while b!=1:
            rr,xx=imp[b];p,q=subtree[b];cost+=rr*p+xx*q;b=parent[b]
        costs[a]=cost
    return F(1)-F(2*max(costs.values()),100000*1266**2)


def prepare(branches,loads,first_layer,limit=512):
    edges={int(r['branch_id']):(int(r['from_bus']),int(r['to_bus'])) for r in branches};nodes=list(range(1,70))
    excluded={tuple(c) for c in first_layer}|{(14,56,61,69,70)}
    pool=set();independent=set()
    for c in first_layer:
        pool.update(cycle_neighbors(nodes,edges,c));independent.update(exhaustive_neighbors(nodes,edges,c))
    pool-=excluded;independent-=excluded;assert pool==independent
    # Unit/scale fixture: 1 MW + j0.5 Mvar through 0.1+j0.2 ohm at 10 kV.
    toy=[dict(branch_id='1',from_bus='1',to_bus='2',r_ohm='0.1',x_ohm='0.2')]
    assert profile([],toy,[dict(bus='2',p_kw='1000',q_kvar='500')],F(10))[2]==F('0.996')
    assert profile([],toy,[],F(10))[2]==1
    scored=[]
    for c in sorted(pool):
        score=min(profile(c,branches,loads).values())
        assert score==integer_score(c,branches,loads)
        scored.append((score,c))
    scored.sort(key=lambda item:(-item[0],item[1]))
    records=[dict(rank=i,open_branches=list(c),minimum_linear_squared_voltage_numerator=v.numerator,
        minimum_linear_squared_voltage_denominator=v.denominator) for i,(v,c) in enumerate(scored,1)]
    return records, records[:limit], dict(status='PASS',full_pool_count=len(pool),selected_count=min(limit,len(pool)),
        independent_all_parent_exchange_enumeration=True,all_scores_fraction_vs_integer_equal=True,
        analytic_unit_fixture=True,zero_load_fixture=True,test_solver_calls=0)
