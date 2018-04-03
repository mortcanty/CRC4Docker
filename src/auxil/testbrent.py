 
from scipy.optimize import minimize_scalar

class Test(object):
    
    def __init__(self,brack):
        self.brack = brack
        
    def theta(self,sigma):
        print sigma
        return (sigma-5.0)**2
    
    def train(self):
        result = minimize_scalar(self.theta,bracket=self.brack)
        if result.success:
            self.sigma_min = result.x
            return True 
        else:
            print result.message
        
tst = Test((1.0,6.0,10.0)) 
if tst.train():
    print 'minimum: %f'%tst.sigma_min      
        