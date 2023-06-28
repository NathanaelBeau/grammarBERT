import ast
import json

from collections import deque
import re

import astor

from .grammar import *

p_elif = re.compile(r'^elif\s?')
p_else = re.compile(r'^else\s?')
p_try = re.compile(r'^try\s?')
p_except = re.compile(r'^except\s?')
p_finally = re.compile(r'^finally\s?')
p_decorator = re.compile(r'^@.*')

def canonicalize_code(code):
    if p_elif.match(code):
        code = 'if True: pass\n' + code

    if p_else.match(code):
        code = 'if True: pass\n' + code

    if p_try.match(code):
        code = code + 'pass\nexcept: pass'
    elif p_except.match(code):
        code = 'try: pass\n' + code
    elif p_finally.match(code):
        code = 'try: pass\n' + code

    if p_decorator.match(code):
        code = code + '\ndef dummy(): pass'

    if code[-1] == ':':
        code = code + 'pass'

    return ast.parse(code)

def make_iterlists(action_sequence):
    # makes lists in the history
    # On fait une liste où on fait rentrer tous les arguments multiples
    result = []
    while action_sequence:
        action, iterflag = action_sequence.popleft()
        if isinstance(action,
                      ReduceAction):
            if iterflag == '-' or iterflag == '*':
                return result
            elif iterflag == '+':
                result.append(([], '-'))
            else:
                result.append((None, '-'))
        elif iterflag == '+':
            result.append(([(action, iterflag)] + make_iterlists(action_sequence), '-'))
        else:
            result.append((action, iterflag))
    return result


def seq2ast(action_sequence, rec_call=False):
    # sequences are sequences of tuples (action,iterflag) or (list,iterflag)
    stack = []
    while action_sequence:
        action, itersymbol = action_sequence.pop()
        if isinstance(action, GrammarRule):
            arity = action.arity()
            ast = action.build_ast(stack[:arity])
            stack = [ast] + stack[arity:]
        elif isinstance(action, list):
            stack = [seq2ast(action, rec_call=True)] + stack
        else:  # push constant value
            stack = [action] + stack
    return stack if rec_call else stack[0]


def ast2seq(tree, action_dict, parent_type=(), parent_field=(), parent_cardinality=(),
            primitives=['identifier', 'int', 'string', 'object', 'constant']):
            # primitives=['identifier', 'int', 'string', 'bytes', 'object', 'singleton', 'constant']):
    action_sequence = []
    if isinstance(tree, ast.AST):
        action = action_dict[tree.__class__.__name__]
        action_sequence.append((action, '-'))
        for i, field in enumerate(tree._fields):
            parent_type += (action.rhs[i],)
            parent_field += (action.rhs_names[i],)
            parent_cardinality += (action.iter_flags[i],)
            # print(parent)
            child = getattr(tree, field)
            extended_actions, parent_type, parent_field, parent_cardinality = ast2seq(child, action_dict, parent_type,
                                                                                      parent_field, parent_cardinality)
            action_sequence.extend(extended_actions)
    elif isinstance(tree, list):  # multiple cardinality
        if tree == []:
            if parent_type[-1] in primitives:
                action_sequence.append((ReduceAction('Reduce_primitif'), '+'))
            else:
                action_sequence.append((ReduceAction('Reduce'), '+'))
        else:
            type_reduce = parent_type[-1]
            field_reduce = parent_field[-1]
            cardinality_reduce = parent_cardinality[-1]
            for idx, arg in enumerate(tree):
                argseq, parent_type, parent_field, parent_cardinality = ast2seq(arg, action_dict, parent_type,
                                                                                parent_field, parent_cardinality)
                action, iterflag = argseq[0]
                argseq[0] = (action, '+') if idx == 0 else (action, '*')
                action_sequence.extend(argseq)
                parent_type += (type_reduce,)
                parent_field += (field_reduce,)
                parent_cardinality += (cardinality_reduce,)
            action_sequence.append((ReduceAction('Reduce'), '-'))
    elif isinstance(tree, type(None)):  # optional cardinality
        if parent_type[-1] in primitives:
            action_sequence.append((ReduceAction('Reduce_primitif'), '?'))
        else:
            action_sequence.append((ReduceAction('Reduce'), '?'))
    else:  # primitives
        action_sequence.append((tree, '-'))
    return action_sequence, parent_type, parent_field, parent_cardinality


def depth_ast(root):
    return 1 + max(map(depth_ast, ast.iter_child_nodes(root)),
               default=0)



if __name__ == '__main__':
    asdl_text = open('./asdl/PythonASDLgrammar3,9.txt').read()
    grammar, _, _ = Grammar.from_text(asdl_text)
    act_list = [GrammarRule(rule.constructor.name, rule.type.name, rule.fields) for rule in grammar]
    assert (len(grammar) == len(act_list))
    Reduce = ReduceAction('Reduce')
    act_dict = dict([(act.label, act) for act in act_list])

    py_ast = '''df.to_csv('c:\\data\\pandas.txt', header=None, index=None, sep=' ', mode='a')'''

    py_ast = ast.parse(py_ast)
    values = ast2seq(py_ast, act_dict, [], [], [])
    a = [(a, flag) for a, flag in values[0]]
    print(len(a))
    # print([a.label if isinstance(a, GrammarRule) else a for a, flag in values[0]])
    print('testetsset' , a)
    a = seq2ast(make_iterlists(deque(a)))

    print(a)

    print(ast.dump(a[0]))
    print(astor.to_source(a[0]).rstrip())

    # dataset_django = open('./dataset/data_django/all.code')
    #
    # depth_django = [depth_ast(canonicalize_code(example.strip())) for example in dataset_django]
    #
    # print(depth_django)
    #
    # print('profondeur maximale des arbres du train set django:', max(depth_django))
    #
    # print('profondeur moyenne des arbres du test set django:', np.mean(depth_django))
    #
    # print('profondeur variance des arbres du  dataset django:', np.var(depth_django))



