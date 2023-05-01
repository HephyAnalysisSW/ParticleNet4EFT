import operator
import functools
import numpy as np

class MSELoss:
    
    def __init__( self, base_points):
        #self.derivatives = derivatives
        self.base_points = base_points

#        # precoumputed base_point_const
#        self.base_point_const = np.array([[ functools.reduce(operator.mul, [point[coeff] if (coeff in point) else 0 for coeff in der ], 1) for der in self.derivatives] for point in self.base_points]).astype('float')
#
#        for i_der, der in enumerate(self.derivatives):
#           if not (len(der)==2 and der[0]==der[1]): continue
#           for i_point in range(len(self.base_points)):
#               self.base_point_const[i_point][i_der]/=2.
#
#        assert np.linalg.matrix_rank(self.base_point_const) == self.base_point_const.shape[0], \
#                  "Base points not linearly independent! Found rank %i for %i base_points" %( np.linalg.matrix_rank(self.base_point_const), self.base_point_const.shape[0])

    def __call__( self, predictions, weights ):
        w0 = weights[:,0]
        return sum( [ (weights[:,0]*(predictions[:,0] + 0.5*base_point*predictions[:,1] - (weights[:,1] + 0.5*base_point*weights[:,2])/weights[:,0])**2).sum() for base_point in self.base_points] )
