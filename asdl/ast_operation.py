import ast

import re
from collections import deque
import astor

from .grammar import *

# Regular expressions to match different Python structures
p_elif = re.compile(r'^elif\s?')
p_else = re.compile(r'^else\s?')
p_try = re.compile(r'^try\s?')
p_except = re.compile(r'^except\s?')
p_finally = re.compile(r'^finally\s?')
p_decorator = re.compile(r'^@.*')


# Canonicalizes code snippets, ensuring they are parsable by ast
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


# Converts action sequences into a list for each iteration step
def make_iterlists(action_sequence):
    result = []
    while action_sequence:
        action, iterflag = action_sequence.popleft()
        if isinstance(action, ReduceAction):
            if iterflag in ['-', '*']:
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


# Rebuilds the AST from a sequence of actions and iteration flags
def seq2ast(action_sequence, rec_call=False):
    stack = []
    while action_sequence:
        action, itersymbol = action_sequence.pop()
        if isinstance(action, GrammarRule):
            arity = action.arity()
            ast_node = action.build_ast(stack[:arity])
            stack = [ast_node] + stack[arity:]
        elif isinstance(action, list):
            stack = [seq2ast(action, rec_call=True)] + stack
        else:
            stack = [action] + stack
    return stack if rec_call else stack[0]


# Converts an AST to a sequence of actions, tracking parent types and fields
def ast2seq(tree, action_dict, parent_type=(), parent_field=(), parent_cardinality=(),
            primitives=['identifier', 'int', 'string', 'object', 'constant']):
    action_sequence = []
    if isinstance(tree, ast.AST):
        action = action_dict[tree.__class__.__name__]
        action_sequence.append((action, '-'))
        for i, field in enumerate(tree._fields):
            parent_type += (action.rhs[i],)
            parent_field += (action.rhs_names[i],)
            parent_cardinality += (action.iter_flags[i],)
            child = getattr(tree, field)
            extended_actions, parent_type, parent_field, parent_cardinality = ast2seq(
                child, action_dict, parent_type, parent_field, parent_cardinality
            )
            action_sequence.extend(extended_actions)
    elif isinstance(tree, list):
        if not tree:
            if parent_type[-1] in primitives:
                action_sequence.append((ReduceAction('Reduce_primitif'), '+'))
            else:
                action_sequence.append((ReduceAction('Reduce'), '+'))
        else:
            for idx, arg in enumerate(tree):
                argseq, parent_type, parent_field, parent_cardinality = ast2seq(
                    arg, action_dict, parent_type, parent_field, parent_cardinality
                )
                action, iterflag = argseq[0]
                argseq[0] = (action, '+') if idx == 0 else (action, '*')
                action_sequence.extend(argseq)
            action_sequence.append((ReduceAction('Reduce'), '-'))
    elif isinstance(tree, type(None)):
        if parent_type[-1] in primitives:
            action_sequence.append((ReduceAction('Reduce_primitif'), '?'))
        else:
            action_sequence.append((ReduceAction('Reduce'), '?'))
    else:
        action_sequence.append((tree, '-'))
    return action_sequence, parent_type, parent_field, parent_cardinality


# Calculates the depth of the AST
def depth_ast(root):
    return 1 + max(map(depth_ast, ast.iter_child_nodes(root)), default=0)


if __name__ == '__main__':
    # Load ASDL grammar
    asdl_text = open('./asdl/ASDL3.8.txt').read()
    grammar, _, _ = Grammar.from_text(asdl_text)
    act_list = [GrammarRule(rule.constructor.name, rule.type.name, rule.fields) for rule in grammar]
    assert len(grammar) == len(act_list)

    # Mapping of action labels
    act_dict = dict([(act.label, act) for act in act_list])

    # Parse Python AST from code
    py_ast = '''df.to_csv('c:\\data\\pandas.txt', header=None, index=None, sep=' ', mode='a')'''
    py_ast = ast.parse(py_ast)

    # Convert AST to action sequence
    action_sequence, _, _, _ = ast2seq(py_ast, act_dict, primitives=['identifier', 'int', 'string', 'object', 'constant']) # Primitives set for Python 3.8

    # Create an iterable list of actions
    action_list = [(action, flag) for action, flag in action_sequence]

    # Convert action sequence back to AST
    restored_ast = seq2ast(make_iterlists(deque(action_list)))

    # Output the original AST, restored AST, and source code
    print(ast.dump(restored_ast[0]))
    print(astor.to_source(restored_ast[0]).rstrip())
