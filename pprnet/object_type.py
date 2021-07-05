""" 
Class ObjectType holds properties of each type of object.

Author: Zhikai Dong
"""

class ObjectType():
    r"""
        ObjectType holds properties of each type of object.

        Init args
        ----------
        type_name: str, name of the type
        class_idx: int, class label of the type
        symmetry_type: str, 'revolution' or 'finite'
        kwarg:
            if symmetry_type == 'revolution':
                lambda_p: scalar
                retoreflection: bool
            if symmetry_type == 'finite':
                lambda_p: List[List[float]] (3, 3)
                G: List[ List[List[float]] (3, 3) ], len(G)==K, objects with K equal poses(finite)
    """
    def __init__(self, type_name, class_idx, symmetry_type, **kwarg):
        assert symmetry_type in ['revolution', 'finite']
        self.type_name = type_name
        self.class_idx = class_idx
        self.symmetry_type = symmetry_type
        if symmetry_type == 'revolution':
            self.lambda_p = kwarg['lambda_p']
            self.retoreflection = kwarg['retoreflection']
        elif symmetry_type == 'finite':
            self.lambda_p = kwarg['lambda_p']
            self.G = kwarg['G']

    def get_properties(self):
        '''
            get args for building PoseLossCalculator
            Returns:
                args: dict
        '''
        args = {
            'symmetry_type': self.symmetry_type,
            'lambda_p': self.lambda_p
        }
        if self.symmetry_type == 'revolution':
            args['retoreflection'] = self.retoreflection
        elif self.symmetry_type == 'finite':
            args['G'] = self.G
        return args

