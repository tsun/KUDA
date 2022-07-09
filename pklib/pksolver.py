import numpy as np
import gurobipy as grb
import random

class PK_solver():
    def __init__(self, N, C, C_ub=[], C_br=[], pk_prior_weight=10.):
        self.N = N  # number of samples
        self.C = C  # number of classes
        self.C_ub = C_ub  # constraints of unary bound
        self.C_br = C_br  # constraints of binary relationship
        self.pk_prior_weight = pk_prior_weight

    # create unary bound constraints
    def create_C_ub(self, cls_probs, uconf=0.):
        ubs = cls_probs * (1 + uconf)
        lbs = cls_probs * (1 - uconf)
        ubs[ubs > 1.0] = 1.0
        lbs[lbs < 0.0] = 0.0
        ubs = (ubs*self.N).tolist()
        lbs = (lbs*self.N).tolist()
        self.C_ub = list(zip(list(range(self.C)), lbs, ubs))

    # create unary bound constraints with noises
    def create_C_ub_noisy(self, cls_probs, uconf=0., noise=0.):
        ubs = cls_probs * (1 + uconf)
        lbs = cls_probs * (1 - uconf)
        bias = (2*np.random.rand(len(cls_probs))-1)*cls_probs*noise
        bias -= bias.mean()
        ubs += bias
        lbs += bias
        ubs[ubs > 1.0] = 1.0
        lbs[lbs < 0.0] = 0.0
        ubs = (ubs*self.N).tolist()
        lbs = (lbs*self.N).tolist()
        self.C_ub = list(zip(list(range(self.C)), lbs, ubs))

    # create binary relationship constraints
    def create_C_br(self, cls_probs, uconf=0.):
        idx = np.argsort(-cls_probs)
        self.C_br = [(idx[c], idx[c+1], (1-uconf)*np.round((cls_probs[idx[c]]-cls_probs[idx[c+1]])*self.N)) for c in range(self.C-1)]

    # create binary relationship constraints with noises
    def create_C_br_noisy(self, cls_probs, uconf=0., noise=0.):
        idx = np.argsort(-cls_probs)
        C=len(idx)
        score = np.arange(C) + (2*np.random.rand(C)-1)*noise + np.random.rand(C)*0.0001
        idd = np.argsort(score)
        idx = idx[idd]
        self.C_br = [(idx[c], idx[c+1], 0) for c in range(self.C-1)]

    # create unary bound constraints from (head) partial classes
    def create_C_ub_partial(self, cls_probs, uconf=0., N=10):
        ubs = cls_probs * (1 + uconf)
        lbs = cls_probs * (1 - uconf)
        ubs[ubs > 1.0] = 1.0
        lbs[lbs < 0.0] = 0.0
        ubs = (ubs*self.N).tolist()
        lbs = (lbs*self.N).tolist()
        self.C_ub = list(zip(list(range(self.C)), lbs, ubs))[:N]

    # create unary bound constraints from (tail) partial classes
    def create_C_ub_partial_reverse(self, cls_probs, uconf=0., N=10):
        ubs = cls_probs * (1 + uconf)
        lbs = cls_probs * (1 - uconf)
        ubs[ubs > 1.0] = 1.0
        lbs[lbs < 0.0] = 0.0
        ubs = (ubs*self.N).tolist()
        lbs = (lbs*self.N).tolist()
        self.C_ub = list(zip(list(range(self.C)), lbs, ubs))[-N:]

    # create unary bound constraints from (random) partial classes
    def create_C_ub_partial_rand(self, cls_probs, uconf=0., N=10):
        ubs = cls_probs * (1 + uconf)
        lbs = cls_probs * (1 - uconf)
        ubs[ubs > 1.0] = 1.0
        lbs[lbs < 0.0] = 0.0
        ubs = (ubs*self.N).tolist()
        lbs = (lbs*self.N).tolist()
        self.C_ub = random.sample(list(zip(list(range(self.C)), lbs, ubs)), k=N)

    # create binary relationship constraints from (head) partial classes
    def create_C_br_partial(self, cls_probs, uconf=0., N=10):
        idx = np.argsort(-cls_probs)
        self.C_br = [(idx[c], idx[c+1], (1-uconf)*np.round((cls_probs[idx[c]]-cls_probs[idx[c+1]])*self.N)) for c in range(self.C-1)][:N]

    # create binary relationship constraints from (tail) partial classes
    def create_C_br_partial_reverse(self, cls_probs, uconf=0., N=10):
        idx = np.argsort(-cls_probs)
        self.C_br = [(idx[c], idx[c+1], (1-uconf)*np.round((cls_probs[idx[c]]-cls_probs[idx[c+1]])*self.N)) for c in range(self.C-1)][-N:]

    # create binary relationship constraints from (random) partial classes
    def create_C_br_partial_rand(self, cls_probs, uconf=0., N=10):
        idx = np.argsort(-cls_probs)
        self.C_br = random.sample([(idx[c], idx[c+1], (1-uconf)*np.round((cls_probs[idx[c]]-cls_probs[idx[c+1]])*self.N)) for c in range(self.C-1)], k=N)


    # solver with smooth regularization
    def solve_soft_knn_cst(self, probs, fix_set=[], fix_labels=[], knn_regs=[]):
        # fix_set and fix_labels are samples with given (pseudo) labels that do not require optimization
        fix_cls_probs = np.eye(self.C)[fix_labels].sum(0)

        # var_set are samples to refine (pseudo) labels
        var_set = list(set(range(self.N)) - set(fix_set))
        Nvar = len(var_set)

        # create an optimization model
        LP = grb.Model(name="Prior Constraint Problem")
        x = {(n, c): LP.addVar(vtype=grb.GRB.BINARY,
                               name="x_{0}_{1}".format(n, c))
             for n in range(Nvar) for c in range(self.C)}

        LP.addConstrs( (grb.quicksum(x[n, c] for c in range(self.C))==1) for n in range(len(var_set)))

        objective = grb.quicksum(x[n, c] * probs[var_set[n], c]
                                 for n in range(Nvar)
                                 for c in range(self.C))

        # add soft constraints of unary bound
        xi_ub = {(c,k): LP.addVar(vtype=grb.GRB.CONTINUOUS, lb=-Nvar, ub=Nvar,
                               name="xi_ub_{0}_{1}".format(c,k))
              for c in range(len(self.C_ub)) for k in range(2)}

        xi_lb = {(c,k): LP.addVar(vtype=grb.GRB.CONTINUOUS, lb=-Nvar, ub=Nvar,
                                name="xi_lb_{0}_{1}".format(c,k))
                 for c in range(len(self.C_ub)) for k in range(2)}

        margin_ub = []
        margin_lb = []
        for i, (c, lb, ub) in enumerate(self.C_ub):
            if ub is not None:
                ub = ub - fix_cls_probs[c]
                margin_ub.append(grb.quicksum(x[n, c] for n in range(Nvar))-ub)
            else:
                margin_ub.append(0.)

            if lb is not None:
                lb = lb - fix_cls_probs[c]
                margin_lb.append( - grb.quicksum(x[n, c] for n in range(Nvar)) + lb)
            else:
                margin_lb.append(0.)


        LP.addConstrs(
            (xi_ub[i, 1] == margin_ub[i] for i in range(len(self.C_ub))), name="slack_ub_0"
        )
        LP.addConstrs(
            (xi_ub[i, 0] == grb.max_(xi_ub[i, 1], 0) for i in range(len(self.C_ub))), name="slack_ub_1"
        )

        LP.addConstrs(
            (xi_lb[i, 1] == margin_lb[i] for i in range(len(self.C_ub))), name="slack_lb_0"
        )
        LP.addConstrs(
            (xi_lb[i, 0] == grb.max_(xi_lb[i, 1], 0) for i in range(len(self.C_ub))), name="slack_lb_1"
        )

        constraint_ub = grb.quicksum(xi_ub[c, 0] for c in range(len(self.C_ub))) + \
                         grb.quicksum(xi_lb[c, 0] for c in range(len(self.C_ub)))

        constraint_ub /= (len(self.C_ub) * 2 + 1e-10)

        # add soft constraints of binary relationship
        margin_br = []
        for (c1, c2, diff) in self.C_br:
            diff = diff - fix_cls_probs[c1] + fix_cls_probs[c2]
            margin_br.append(
                -grb.quicksum(x[n, c1] for n in range(Nvar)) + grb.quicksum(x[n, c2] for n in range(Nvar)) + diff)

        xi_br = {(c, k): LP.addVar(vtype=grb.GRB.CONTINUOUS, lb=-2 * Nvar, ub=2 * Nvar,
                                    name="xi_br_{0}_{1}".format(c, k))
                  for c in range(len(self.C_br)) for k in range(2)}

        LP.addConstrs(
            (xi_br[i, 1] == margin_br[i] for i in range(len(self.C_br))), name="slack_br_0"
        )
        LP.addConstrs(
            (xi_br[i, 0] == grb.max_(xi_br[i, 1], 0) for i in range(len(self.C_br))), name="slack_br_1"
        )

        constraint_br = grb.quicksum(xi_br[c, 0] for c in range(len(self.C_br)))
        constraint_br /= (len(self.C_br) + 1e-10)

        constraint = constraint_ub + constraint_br

        # add smooth regularization
        # currently it does NOT support fixset
        if len(knn_regs) > 0:
            LP.addConstrs(
                    (x[knn_regs[i][0], c] == x[knn_regs[i][1][k], c]
                     for i in range(len(knn_regs))
                     for k in range(len(knn_regs[i][1]))
                     for c in range(self.C) ), name="smooth_regularization"
                )

        LP.ModelSense = grb.GRB.MAXIMIZE
        LP.setObjective(objective - self.pk_prior_weight*constraint*Nvar)

        LP.optimize()

        # get refined (pseudo) labels from optimal solution
        var_labels = []
        for n in range(Nvar):
            for c in range(self.C):
                var_labels.append(x[n, c].X)

        var_labels = np.array(var_labels)
        var_labels = var_labels.reshape([Nvar, self.C])
        var_labels = np.argmax(var_labels, axis=-1)

        labels = np.zeros(self.N).astype(np.int32)
        labels[fix_set] = fix_labels
        labels[var_set] = var_labels

        return labels

    # solver without smooth regularization
    def solve_soft(self, probs, fix_set=[], fix_labels=[]):
        fix_cls_probs = np.eye(self.C)[fix_labels].sum(0)

        var_set = list(set(range(self.N)) - set(fix_set))
        Nvar = len(var_set)

        LP = grb.Model(name="Prior Constraint Problem")
        x = {(n, c): LP.addVar(vtype=grb.GRB.BINARY,
                               name="x_{0}_{1}".format(n, c))
             for n in range(Nvar) for c in range(self.C)}

        LP.addConstrs( (grb.quicksum(x[n, c] for c in range(self.C))==1) for n in range(len(var_set)))

        objective = grb.quicksum(x[n, c] * probs[var_set[n], c]
                                 for n in range(Nvar)
                                 for c in range(self.C))

        # add soft constraints of unary bound
        xi_ub = {(c,k): LP.addVar(vtype=grb.GRB.CONTINUOUS, lb=-Nvar, ub=Nvar,
                               name="xi_ub_{0}_{1}".format(c,k))
              for c in range(len(self.C_ub)) for k in range(2)}

        xi_lb = {(c,k): LP.addVar(vtype=grb.GRB.CONTINUOUS, lb=-Nvar, ub=Nvar,
                                name="xi_lb_{0}_{1}".format(c,k))
                 for c in range(len(self.C_ub)) for k in range(2)}

        margin_ub = []
        margin_lb = []
        for i, (c, lb, ub) in enumerate(self.C_ub):
            if ub is not None:
                ub = ub - fix_cls_probs[c]
                margin_ub.append(grb.quicksum(x[n, c] for n in range(Nvar))-ub)
            else:
                margin_ub.append(0.)

            if lb is not None:
                lb = lb - fix_cls_probs[c]
                margin_lb.append( - grb.quicksum(x[n, c] for n in range(Nvar)) + lb)
            else:
                margin_lb.append(0.)


        LP.addConstrs(
            (xi_ub[i, 1] == margin_ub[i] for i in range(len(self.C_ub))), name="slack_ub_0"
        )
        LP.addConstrs(
            (xi_ub[i, 0] == grb.max_(xi_ub[i, 1], 0) for i in range(len(self.C_ub))), name="slack_ub_1"
        )

        LP.addConstrs(
            (xi_lb[i, 1] == margin_lb[i] for i in range(len(self.C_ub))), name="slack_lb_0"
        )
        LP.addConstrs(
            (xi_lb[i, 0] == grb.max_(xi_lb[i, 1], 0) for i in range(len(self.C_ub))), name="slack_lb_1"
        )

        constraint_ub = grb.quicksum(xi_ub[c,0] for c in range(len(self.C_ub))) + \
                     grb.quicksum(xi_lb[c,0] for c in range(len(self.C_ub)))

        constraint_ub /= (len(self.C_ub)*2 + 1e-10)

        # add soft constraints of binary relationship
        margin_br = []
        for (c1, c2, diff) in self.C_br:
            diff = diff - fix_cls_probs[c1] + fix_cls_probs[c2]
            margin_br.append(-grb.quicksum(x[n, c1] for n in range(Nvar)) + grb.quicksum(x[n, c2] for n in range(Nvar)) + diff)

        xi_br = {(c, k): LP.addVar(vtype=grb.GRB.CONTINUOUS, lb=-2*Nvar, ub=2*Nvar,
                                   name="xi_br_{0}_{1}".format(c, k))
                 for c in range(len(self.C_br)) for k in range(2)}

        LP.addConstrs(
            (xi_br[i, 1] == margin_br[i] for i in range(len(self.C_br))), name="slack_br_0"
        )
        LP.addConstrs(
            (xi_br[i, 0] == grb.max_(xi_br[i, 1], 0) for i in range(len(self.C_br))), name="slack_br_1"
        )

        constraint_br = grb.quicksum(xi_br[c, 0] for c in range(len(self.C_br)))
        constraint_br /= (len(self.C_br) + 1e-10)

        constraint = constraint_ub + constraint_br


        LP.ModelSense = grb.GRB.MAXIMIZE
        LP.setObjective(objective - self.pk_prior_weight*constraint*Nvar)

        LP.optimize()

        # get refined (pseudo) labels from optimal solution
        var_labels = []
        for n in range(Nvar):
            for c in range(self.C):
                var_labels.append(x[n, c].X)

        var_labels = np.array(var_labels)
        var_labels = var_labels.reshape([Nvar, self.C])
        var_labels = np.argmax(var_labels, axis=-1)

        labels = np.zeros(self.N).astype(np.int32)
        labels[fix_set] = fix_labels
        labels[var_set] = var_labels

        return labels

