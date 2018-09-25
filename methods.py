"""
Joseph D. Taylor
Department of Physics 
University of Bath, United Kingdom
j.taylor@bath.ac.uk

June 7th 2018

This package generates C++ files required for the
solution of state and parameter estimation problems in the
open-source large-scale ​nonlinear optimization software
IPOPT. Equations of motion of user-defined dynamical systems 
are temporally discretized, and the Jacobian and Hessian 
matrices of the discrete system are calculated with symbolic 
differentiation. The equations of motion are imposed as 
strong constraints on the minimization of a cost function
defining the distance between the model and time series 
observations of the system to be estimated.

This code requires the installation of the following:
    
    - Sympy (.py)
    - Numpy (.py)
    - IPOPT 
    
IPOPT can be downloaded from https://projects.coin-or.org/Ipopt

This software package was used in the following publications:
    
    [1] J. Taylor et al., Stochasticity and convergence in data 
        assimilation of predictive neuron models
        
    [2] K.Abu-Hassan et al., Construction of neuromorphic models 
        of respiratory neurons

"""

import re
import sympy as sym
from itertools import groupby


class DynamicalSystem(object):
    
    def __init__(self):
        """
        Constructor for the Dynamical System class.
        """
        self.init_path = None
    
    def define_system(self, filename, nX, nP):
        """
        Define dynamical system, from text file optional.
        nX and nP are the dimensions of the system and the
        set of system parameters, respectively.
        """
        self.nX = nX
        self.nP = nP
        
        content = []
        for line in open(filename):
            ls = line.strip()
            if not ls.startswith('#'):
                content.append(ls)
        
        self.sVars = content[0:nX]        
        self.sPars = content[nX:nX+nP]
        self.sEqns = content[nX+nP:2*nX+nP]
        
    
    def set_bounds(self, path='../data/bounds.txt', lower=None, upper=None):
        """
        Define path for bounds file.
        """
        self.bounds_path = path
        
        if lower != None:
            self.bound_l = lower
        if upper != None:
            self.bound_u = upper

    
    def set_observations(self, idx_meas, idx_ctrl, *data_tuple):
        """
        Set observed and controlled variables by placing
        indices in idx_meas and idx_ctrl. Control indices
        must be a subset of the measured variables.
        
        Define data name, data path, control name
        (optional) in n-tuple for *data_tuple.
        """
        obs_temp = []         
        ctrl_temp = []
        for arg in data_tuple:
            obs_temp.append(arg)
            if len(arg) > 2:
                ctrl_temp.append(arg[2])
                ctrl_temp.append('d' + arg[2])

        if len(obs_temp) != len(idx_meas):
            print('Error: dimensions of measurement function and declared data do not match.')
        elif set(idx_ctrl).issubset(idx_meas) == False:
            print('Error: controlled variables must be a subset of observed variables.')
        else:
            self.obs = obs_temp
            self.sCtrl = ctrl_temp

        self.idx_meas = idx_meas
        self.idx_ctrl = idx_ctrl                    
                    
        
    def set_forcing(self, *data_tuple):
        """
        Set forcing functions for dynamical system.
        Define data name and data path in tuple.        
        """
        forc_temp = []
        for arg in data_tuple:
            if len(arg) != 2:
                print('Error: please define *data_tuple as pair (data_name, data_path).')
            else:
                forc_temp.append(arg)            
        self.forc = forc_temp
        
        
    def set_initial(self, path=None, ctrl=0):
        """
        Set initial state for assimiliation model.
        """ 
        self.init_path = path
        self.n_init = len(self.sVars)
        if ctrl != 0:
            self.ctrl_init = ctrl
        else: 
            self.ctrl_init = 0

    
    def simpsonify(self, hermite=False):
        """
        Reads in continuous-time system equations, sEqns,
        and transforms them into discrete-time mappings,
        dEqns, according to Simpson's rule.
        
        Hermite polynomial constraints are optional, 
        but prevent oscillatory solutions which fluctuate
        about the true state history.
        
        The final constraint enforces smooth evolution of
        the control terms over the observation window.
        """
        self.dEqns = []
        self.tp_val = 3
        self.dVars = [v + '_t%d' % x for v in self.sVars for x in range(3)]
        self.dCtrl = [v + '_t%d' % x for v in self.sCtrl for x in range(3)]
        for eqn in range(len(self.sEqns)):
            output = []
            self.dDict = {}
            for time_point in range(3):
                for var in range(len(self.sVars)):
                    self.dDict.update({self.sVars[var] : self.dVars[3*var+time_point]})
                for ctrl in range(len(self.sCtrl)):
                    self.dDict.update({self.sCtrl[ctrl] : self.dCtrl[3*ctrl+time_point]})
                output.append(re.sub(r'\b' + '|'.join(self.dDict.keys()) + r'\b', 
                            lambda m: self.dDict[m.group(0)], self.sEqns[eqn]))
            
            self.dEqns.append('%s_t2 - %s_t0 - 2*h_*((%s)+4*(%s)+(%s))/6.0' % (self.sVars[eqn],
                              self.sVars[eqn], output[0], output[1], output[2]))
        
            if hermite == True:
                self.dEqns.append('%s_t1 - (%s_t0 + %s_t2)/2.0 - 2*h_*((%s) - (%s))/8.0' % 
                                  (self.sVars[eqn], self.sVars[eqn], self.sVars[eqn], 
                                   output[0], output[2]))
                
        for ctrl in range(len(self.sCtrl)):
            if ctrl % 2 == 0:
                self.dEqns.append('%s_t1 - (%s_t0 + %s_t2)/2.0 - 2*h_*(%s_t0 - %s_t2)/8.0' % 
              (self.sCtrl[ctrl], self.sCtrl[ctrl], self.sCtrl[ctrl], 
              self.sCtrl[ctrl+1], self.sCtrl[ctrl+1]))
            
                
                
    def objective_function(self, method='least_squares', controls=True):
        """
        Define form of cost functional to be minimized
        during variational inference. Default settings
        correspond to a least-squares cost function
        consisting of all observed variables and any 
        corresponding control terms.
        
        This function must be updated for weak-constraint
        4D-var, including the addition of covariance 
        matrices to the cost function.
        """
        obj_func = ''
        for data in range(len(self.obs)):
            var_idx = self.idx_meas[data]
            obj_func = obj_func + '+(%s-%s)*(%s-%s)' % (self.obs[data][0], self.sVars[var_idx],
                                    self.obs[data][0], self.sVars[var_idx])
            if len(self.obs[data]) > 2:
                obj_func = obj_func + '+%s*%s' % (self.obs[data][2], self.obs[data][2])
        
        self.obj_func = obj_func
        
    
    def differentiate_objective(self):
        """
        Calculates the (double) derivatives of the 
        objective function with respect to state 
        variables, data, and controls. 
        
        Non-zero gradient values are stored in an 
        array with the format:
            
            [var_idx, 'nnz_value']
        
        while non-zero hessian values are stored as:
            
            [var_1 idx, var_2 idx, 'nnz_value']
        """
        count = 0
        count2 = 0
        output = []
        output2 = []
        var_array = []
        expr = sym.factor(sym.sympify(self.obj_func))
        for var in range(len(self.sVars)):
            var_array.append(sym.symbols(self.sVars[var]))
        for ctrl in range(len(self.sCtrl)):
            var_array.append(sym.symbols(self.sCtrl[ctrl]))
        for var in range(len(self.sVars+self.sCtrl)):
            deriv = (expr).diff(var_array[var])
            if str(deriv) != '0':
                temp = []
                output.append(temp)
                output[count].append(var)
                output[count].append(str(sym.ccode(deriv)))
                count += 1
                for var2 in range(len(var_array)):
                    double_deriv = deriv.diff(var_array[var2])
                    if str(double_deriv) != '0':
                        temp = []
                        output2.append(temp)
                        output2[count2].append(var)
                        output2[count2].append(var2)
                        output2[count2].append(str(sym.ccode(double_deriv)))
                        count2 += 1
        
        self.grad_obj = output
        self.hess_obj = output2
        
        
    def get_ipopt_hessian(self):
        self.ipopt_jacobian = []
        self.ipopt_hessian = []
        self.objective_function()
        self.differentiate_objective()
        self.tVars = self.sVars + self.sCtrl + self.sPars
        self.dtVars = self.dVars + self.dCtrl + self.sPars
        for con in self.dEqns:
            count = 0
            count2 = 0
            output = []
            hess_mat = []
            var_array = []
            hess_register = []
            expr = sym.sympify(con)
            for var in range(len(self.dtVars)):
                var_array.append(sym.symbols(self.dtVars[var]))
            for var in range(len(var_array)):
                jac = expr.diff(var_array[var])
                for var2 in range(len(var_array)):
                    if tuple(sorted((var, var2))) not in hess_register:   
                        hess = str(sym.ccode(jac.diff(var_array[var2])))
                        if hess != '0':
                            temp = []
                            hess_mat.append(temp)
                            hess_mat[count2].append(var)
                            hess_mat[count2].append(var2)
                            hess_mat[count2].append(hess)
                            idx = self.dtVars[var].split('_', 1)[0]
                            var_idx = self.tVars.index(idx)
                            idx2 = self.dtVars[var2].split('_', 1)[0]
                            var_idx2 = self.tVars.index(idx2)
                            hess_mat[count2].append((var_idx, var_idx2))
                            hess_register.append(tuple(sorted((var,var2))))
                            count2 += 1   
                jac = str(sym.ccode(jac))
                if jac != '0':
                    temp = []
                    output.append(temp)
                    output[count].append(var)
                    output[count].append(jac)
                    count += 1
            self.ipopt_jacobian.append(output)
            self.ipopt_hessian.append(hess_mat)
        self.jacobian_structure()
        self.hessian_structure()
        self.ipopt_matrices_nnz()
            
            
    def jacobian_structure(self):
        idx = 0
        jac_struct = []
        for con in range(len(self.dEqns)):
            jac_struct.append([])
            temp = []
            for ele in range(len(self.ipopt_jacobian[con])):
                temp.append(self.ipopt_jacobian[con][ele][0])
            for var in range(len(self.sVars + self.sCtrl)):
                for tp in range(self.tp_val):
                    if ((var*self.tp_val + tp) in temp):
                        jac_struct[idx].append("j + %d+%d*%s" % ((tp),var,"nObs_"))
            for par in range(len(self.sPars)):
                if ((len(self.sVars + self.sCtrl) * self.tp_val + par) in temp):
                    jac_struct[idx].append("%s*%d+%d" % ("nObs_", len(self.sVars + self.sCtrl), par))
            idx += 1
        self.ipopt_jac_struct = jac_struct   
            

    def hessian_structure(self):
        temp = []
        temp2 = []
        nnz_coord = []
        par_count = 0
        nnz_coord_zip = []
        nnz_coord_obj = []
        nnz_nopar_coord = []
        hess_struct = []
        new_coord_idx = 0
        for var in range(len(self.hess_obj)):
            temp2 = []
            temp2.append('j + %d*nObs_' % self.hess_obj[var][0])
            temp2.append('j + %d*nObs_' % self.hess_obj[var][1])
            temp.append(temp2)
            for tp in range(self.tp_val):
                hv = self.hess_obj[var]
                nnz_coord_obj.append((hv[0]*self.tp_val+tp,hv[1]*self.tp_val+tp))
        hess_struct.append(temp)
        for con in range(len(self.ipopt_hessian)):
            self.ipopt_hessian[con].sort(key=lambda x: x[3])
            self.ipopt_hessian[con] = [list(v) for k,v in groupby(self.ipopt_hessian[con], lambda x: x[3])]
            temp3 = []
            for group in range(len(self.ipopt_hessian[con])):
                temp2 = []
                for ele in range(len(self.ipopt_hessian[con][group])):
                    temp = []
                    coordinates = (self.ipopt_hessian[con][group][ele][0],self.ipopt_hessian[con][group][ele][1])
                    if coordinates in nnz_coord_obj + nnz_nopar_coord:
                        temp.append(1)
                        loc = (nnz_coord_obj+nnz_nopar_coord).index(coordinates)
                        temp.append(int(loc/self.tp_val))
                    else:
                        temp.append(0)
                        temp.append(int(new_coord_idx/self.tp_val))
                        if len(self.ipopt_hessian[con][group]) > 1:
                            new_coord_idx += 1
                            nnz_nopar_coord.append(coordinates)
                        nnz_coord.append(coordinates)
                        nnz_coord_zip.append([coordinates,par_count])
                        if len(self.ipopt_hessian[con][group]) == 1:
                            par_count += 1
                    if len(self.ipopt_hessian[con][group]) > 1:
                        if self.ipopt_hessian[con][group][ele][0] < self.tp_val*len(self.sVars + self.sCtrl):
                            temp.append('j + %d*nObs_' % int((self.ipopt_hessian[con][group][ele][0])/self.tp_val))
                        else:
                            temp.append('%d*nObs_+%d' % (len(self.sVars + self.sCtrl),int(self.ipopt_hessian[con][group][ele][3][0] - len(self.sVars + self.sCtrl))))
                        if self.ipopt_hessian[con][group][ele][1] < self.tp_val*len(self.sVars + self.sCtrl):
                            temp.append('j + %d*nObs_' % int((self.ipopt_hessian[con][group][ele][1])/self.tp_val))
                        else:
                            temp.append('%d*nObs_+%d' % (len(self.sVars + self.sCtrl),int(self.ipopt_hessian[con][group][ele][3][1] - len(self.sVars + self.sCtrl))))
                        temp2.append(temp)
                    else:
                        temp.append('%d*nObs_+%d' % (len(self.sVars + self.sCtrl), self.ipopt_hessian[con][group][ele][0] - len(self.sVars + self.sCtrl)*self.tp_val))
                        temp.append('%d*nObs_+%d' % (len(self.sVars + self.sCtrl), self.ipopt_hessian[con][group][ele][1] - len(self.sVars + self.sCtrl)*self.tp_val))
                        temp.append('%d' % (self.ipopt_hessian[con][group][ele][0] - len(self.sVars + self.sCtrl)*self.tp_val))
                        temp.append(nnz_coord_zip[nnz_coord.index(coordinates)][1])
                        temp2.append(temp)
                temp3.append(temp2)
            hess_struct.append(temp3)
        self.ipopt_hess_struct = hess_struct

        
    def ipopt_matrices_nnz(self):
        nnz_jac_g = 0
        for var in range(len(self.ipopt_jac_struct)):
            for ele in range(len(self.ipopt_jac_struct[var])):
                nnz_jac_g += 1
                
        nnz_h_obj = 0
        nnz_h_con = 0
        nnz_h_par = 0
        for var in range(len(self.ipopt_hess_struct[0])):
            nnz_h_obj += 1
        for con in range(1, len(self.ipopt_hess_struct)):
            for group in range(len(self.ipopt_hess_struct[con])):
                    if self.ipopt_hess_struct[con][group][0][0] == 0:
                        if len(self.ipopt_hess_struct[con][group]) > 1:
                            nnz_h_con += 1
                        else:
                            nnz_h_par += 1
                    
        self.nnz_jac_g = nnz_jac_g
        self.nnz_h_obj = nnz_h_obj
        self.nnz_h_con = nnz_h_con
        self.nnz_h_par = nnz_h_par
        