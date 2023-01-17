from collections import UserList
import re
from warnings import warn
from typing import Callable, Any, NoReturn, Iterable, Sequence, Tuple

class Pipeline(UserList):
    """
    Example
    -------
    >>> from skimage import io
    >>> from skimage.morphology import white_tophat, remove_small_holes, remove_small_objects, binary_dilation, disk
    >>> def imthresh(image, value):
    ...     return image < value
    >>> def bit_xor(ar1, ar2):
    ...     return ar1 ^ ar2
    >>> defect_surrounding = [
    ...     {
    ...         'function': io.imread,
    ...         'checkpoint': 'loaded',
    ...     }, {
    ...         'function': imthresh,
    ...         'args':('__', 50),
    ...         'checkpoint': 'thresholded'
    ...     }, {
    ...         'function': white_tophat,
    ...     }, {
    ...         'function': bit_xor,
    ...         'args': ('__', '$$thresholded$$')
    ...     }, {
    ...         'function': remove_small_objects,
    ...         'args': ('__', 80)
    ...     }, {
    ...         'function': remove_small_holes,
    ...         'args': ('__', 400),
    ...         'checkpoint': 'defect_mask'
    ...     }, {
    ...         'function': binary_dilation,
    ...         'args': ('__', disk(20))
    ...     }, {
    ...         'function': bit_xor,
    ...         'args': ('__', '$$defect_mask$$'),
    ...         'checkpoint': 'surrounding_mask'
    ...     }
    ... ]
    >>> defect_surrounding_processor = Pipeline(defect_surrounding)
    >>> defect_surrounding_processor('./data/Inclusion/Inclusion3.png')
    """
    def __init__(self, *args, **kwargs):
        super(Pipeline, self).__init__(*args, **kwargs)
        self.checkpoints = dict()  # Storage for checkpoints.
        self.result = None  # Computation result of the pipeline.
        self.parent_pipeline = None  # If specified, initial value will be pulled from a pipeline specified in here.
        self.starting_checkpoint = None  # If specified, initial value will be pulled from a checkpoint. Else, from result.

    def __call__(self, data:Any = None) -> Any:
        # Initialisation
        self.result = None
        self.checkpoints = dict()

        if (data is None) and (self.parent_pipeline is None):
            raise ValueError("Please provide a data will be processed.")
        elif (data is not None) and (self.parent_pipeline is not None):
            warn("Input data and parent are either specified. Input data is being ignored. "
                 "Further calculation will be performed with the data which pulled from the "
                 "parent pipeline.")

        if self.parent_pipeline is not None:  # If parent is specified.
            if self.starting_checkpoint is not None:  # If checkpoint is specified.
                try:
                    data = self.parent_pipeline[self.starting_checkpoint]  # Pull from the checkpoint of parent pipeline
                except KeyError:
                    raise KeyError(f"A requested checkpoint `{self.starting_checkpoint}` is not available. "
                                   f"Please make sure the parent pipeline is executed first.")
            else:
                data = self.parent_pipeline.result  # Pull from the result of parent pipeline.
        return self._execute(data)

    def checkpoint(self, name: str, value: Any) -> NoReturn:
        """
        Set checkpoint with given name and value.

        Parameters
        ----------
        name: str
        value: *

        Returns
        -------

        """
        reserved = ('',)  # Reserved keywords.
        if name in reserved:
            raise ValueError(f"Your desired checkpoint name {name} is reserved. Please select other.")
        if name in self.checkpoints.keys():
            raise KeyError(f"Your desired checkpoint name '{name}' is already occupied. Please select other.")
        self.checkpoints.update({name: value})

    def append(self, items: Sequence) -> NoReturn:
        for item in items:
            if not isinstance(item, dict):
                raise ValueError("A content of the provided item should be dictionary.")
            keys = item.keys()
            for key in keys:
                if key in ('function', 'args', 'kwargs', 'checkpoint', 'result_select'):
                    pass
                else:
                    raise ValueError("Allowed keys of the provided dictionary is "
                                     "`function`, `args`, `result_select`, `checkpoint`, and `kwargs`.")
        if len(items) == 1:
            super(Pipeline, self).append(items)
        else:
            for item in items:
                super(Pipeline, self).append(item)

    def select_result(self, result:Tuple[Any], index:int) -> Any:
        """
        Selecting result by the provided index. For example, an arbitrary function `f(x) -> a, b` return values are
        not unity. A value selection is required due to the ambiguity. This function is triggered when `select_value`
        key is specified in the directive block.

        Parameters
        ----------
        result : tuple
        index : int

        Returns
        -------
        any

        """
        return result[index]

    def result_redirector(self, result:Any, args:tuple) -> tuple:
        new_args = tuple(result if arg == '__' else arg for arg in args)
        return tuple(new_args)

    def checkpoint_redirector(self, args:tuple) -> tuple:
        new_args = list(args)
        for idx, arg in enumerate(args):
            try:  # Try match
                checkpoint_name = re.match("^\$\$(.+)\$\$$", arg).groups()[0]
                new_args[idx] = self.checkpoints[checkpoint_name]
            except (AttributeError, TypeError):  # if match failed.
                pass
        return tuple(new_args)

    def _get(self, dictionary:dict, key:str) -> Any:
        try:
            value = dictionary[key]
        except KeyError:
            value = None
        return value

    def attach(self, other_pipeline, checkpoint_name=None) -> NoReturn:
        """
        Set parent pipeline.

        Parameters
        ----------
        other_pipeline: Pipeline
            Other `Pipeline` object.
        checkpoint_name: str, optional
            If specified then pull the value from the specific checkpoint of the parent pipeline, which specified by
            `other_pipeline` argument.
        """
        self.from_checkpoint = True
        self.parent_pipeline = other_pipeline
        self.starting_checkpoint = checkpoint_name

    def detach(self) -> NoReturn:
        """
        Reset parent pipeline information.
        """
        self.from_checkpoint = False
        self.parent_pipeline = None
        self.starting_checkpoint = None

    def _execute(self, result):
        for order, directive in enumerate(self):
            # Get values for the result redirection.
            func: Callable = directive['function']  # Mandatory, callable.

            args = self._get(directive, 'args')
            if args is None:
                args = ('__',)
            kwargs = self._get(directive, 'kwargs')
            checkpoint = self._get(directive, 'checkpoint')
            result_select = self._get(directive, 'result_select')

            # Result redirection & checkpoint values get
            new_args = self.result_redirector(result, args)
            new_args = self.checkpoint_redirector(new_args)

            if (kwargs == '') or (kwargs is None):
                result = func(*new_args)
            else:
                result = func(*new_args, **kwargs)

            if checkpoint is not None:
                self.checkpoint(checkpoint, result)

            if result_select is not None:
                result = (result[i] for i in result_select)
        self.result = result
        return result
