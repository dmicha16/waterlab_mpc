import casadi as ca

# opts = dict(printLevel='1') # {'qpoases':{'printLevel':0}}

opts = {"printLevel": 0}
mpc_solver = ca.qpsol('mpc_solver', 'qpoases', {}, opts)