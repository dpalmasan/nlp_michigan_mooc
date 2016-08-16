class Transition(object):
    """
    This class defines a set of transitions which are applied to a
    configuration to get the next configuration.
    """
    # Define set of transitions
    LEFT_ARC = 'LEFTARC'
    RIGHT_ARC = 'RIGHTARC'
    SHIFT = 'SHIFT'
    REDUCE = 'REDUCE'

    def __init__(self):
        raise ValueError('Do not construct this object!')

    @staticmethod
    def left_arc(conf, relation):
        """
            :param configuration: is the current configuration
            :return : A new configuration or -1 if the pre-condition is not satisfied
        """
        if not conf.buffer or not conf.stack:
            return -1

        # s is the node on top of the stack
        s = conf.stack[-1]

        if s == 0:
            return -1

        has_head = False
        
        for arc in conf.arcs:
            if arc[2] == s:
                return -1

        # We pop the stack
        conf.stack.pop(-1)

        # b is the first node in the buffer
        b = conf.buffer.pop(0)
        conf.arcs.append((b, relation, s))

    @staticmethod
    def right_arc(conf, relation):
        """
            :param configuration: is the current configuration
            :return : A new configuration or -1 if the pre-condition is not satisfied
        """
        if not conf.buffer or not conf.stack:
            return -1

        # You get this one for free! Use it as an example.

        #From the paper
        # s is on top of the stack
        idx_wi = conf.stack[-1]

        # b is the first node on the buffer (this is why we pop 0)
        idx_wj = conf.buffer.pop(0)

        # We push the node b onto the stack
        conf.stack.append(idx_wj)

        # We add the arc (s, l, b) to A
        conf.arcs.append((idx_wi, relation, idx_wj))

    @staticmethod
    def reduce(conf):
        """
            :param configuration: is the current configuration
            :return : A new configuration or -1 if the pre-condition is not satisfied
        """
        
        if not conf.buffer or not conf.stack:
            return -1

        has_head = False
        for arc in conf.arcs:
            if arc[2] == conf.stack[-1]:
                has_head = True
        
        if has_head:
            conf.stack.pop(-1)
        else:
            return -1


    @staticmethod
    def shift(conf):
        """
            :param configuration: is the current configuration
            :return : A new configuration or -1 if the pre-condition is not satisfied
        """
        if not conf.buffer or not conf.stack:
            return -1

        conf.stack.append(conf.buffer.pop(0))
