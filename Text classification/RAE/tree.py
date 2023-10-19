

class RTree(object):
    """docstring for RTree."""
    def __init__(self,
                 x=-1,
                 K=-1,#softmax class
                 loss=-1,
                 lc=None,
                 rc=None,):
        super(RTree, self).__init__()
        self.x=x
        self.K=K
        self.loss=loss
        self.lc=lc
        self.rc=rc
        self.leave_count=0
        
    def cal_leaves(self,root=-1):
        if root == -1:
            self.leave_count=0
            self.cal_leaves(self.lc)
            self.cal_leaves(self.rc)
        else:
            if root == None:
                return
            if root.lc == None and root.rc == None:
                self.leave_count+=1
            self.cal_leaves(root.lc)
            self.cal_leaves(root.rc)
            

    