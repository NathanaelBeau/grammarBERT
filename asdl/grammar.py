import sys
from collections import OrderedDict
from itertools import chain
from .utils import remove_comment


class Grammar:
    def __init__(self, productions):
        # productions are indexed by their head types
        self.cardinalities = ['+', '*', '-', '?']
        self._productions = OrderedDict()
        self._constructor_production_map = dict()
        for prod in productions:
            if prod.type not in self._productions:
                self._productions[prod.type] = list()
            self._productions[prod.type].append(prod)
            self._constructor_production_map[prod.constructor.name] = prod

        self.root_type = productions[0].type
        # number of constructors
        self.size = sum(len(head) for head in self._productions.values())

        fields = list(set([field.name for field in self.fields]))
        fields.sort()

        # get entities to their ids map
        self.type2id = OrderedDict({type.name: i for i, type in enumerate(self.types)})
        self.field2id = OrderedDict({field: i for i, field in enumerate(fields)})
        self.cardinality2id = OrderedDict({cardinality: i for i, cardinality in enumerate(self.cardinalities)})

        self.id2type = OrderedDict({i: type.name for i, type in enumerate(self.types)})
        self.id2field = OrderedDict({i: field for i, field in enumerate(fields)})
        self.id2cardinality = OrderedDict({i: cardinality for i, cardinality in enumerate(self.cardinalities)})

    def __len__(self):
        return self.size

    @property
    def productions(self):
        return sorted(chain.from_iterable(self._productions.values()), key=lambda x: repr(x))

    def __getitem__(self, datum):
        if isinstance(datum, str):
            return self._productions[ASDLType(datum)]
        elif isinstance(datum, ASDLType):
            return self._productions[datum]

    def get_prod_by_ctr_name(self, name):
        return self._constructor_production_map[name]

    @property
    def types(self):
        if not hasattr(self, '_types'):
            all_types = set()
            for prod in self.productions:
                all_types.add(prod.type)
                all_types.update(map(lambda x: x.type, prod.constructor.fields))

            self._types = sorted(all_types, key=lambda x: x.name)

        return self._types

    @property
    def fields(self):
        if not hasattr(self, '_fields'):
            all_fields = set()
            for prod in self.productions:
                all_fields.update(prod.constructor.fields)

            self._fields = sorted(all_fields, key=lambda x: (x.name, x.type.name, x.cardinality))

        return self._fields

    @property
    def primitive_types(self):
        return filter(lambda x: isinstance(x, ASDLPrimitiveType), self.types)

    @property
    def composite_types(self):
        return filter(lambda x: isinstance(x, ASDLCompositeType), self.types)

    def is_composite_type(self, asdl_type):
        return asdl_type in self.composite_types

    def is_primitive_type(self, asdl_type):
        return asdl_type in self.primitive_types

    @staticmethod
    def from_text(text):
        def _parse_field_from_text(_text):
            d = _text.strip().split(' ')
            name = d[1].strip()
            type_str = d[0].strip()
            cardinality = '-'
            if type_str[-1] == '*':
                type_str = type_str[:-1]
                cardinality = '+'
            elif type_str[-1] == '?':
                type_str = type_str[:-1]
                cardinality = '?'

            if type_str in primitive_type_names:
                return Field(name, ASDLPrimitiveType(type_str), cardinality=cardinality)
            else:
                return Field(name, ASDLCompositeType(type_str), cardinality=cardinality)

        def parse_field(text):
            text = text.strip()
            if '(' in text:
                field_blocks = text[text.find('(') + 1:text.find(')')].split(',')
                d = field_blocks[0].strip().split(' ')
                name = d[1].strip()
                return name

        def _parse_constructor_from_text(_text):
            _text = _text.strip()
            fields = None
            if '(' in _text:
                name = _text[:_text.find('(')]  # collect left part (constructor name)
                field_blocks = _text[_text.find('(') + 1:_text.find(')')].split(',')  # collect all fields
                fields = map(_parse_field_from_text, field_blocks)
            else:
                name = _text

            return ASDLConstructor(name, fields)

        lines = remove_comment(text).split('\n')
        lines = list(map(lambda l: l.strip(), lines))  # remove blank

        line_no = 0

        # first line is always the primitive types
        primitive_type_names = list(map(lambda x: x.strip(), lines[line_no].split(',')))
        lines = lines[1:]
        # Clean raw text from the grammar
        index = 1
        fields = []
        types = []

        while index < len(lines):
            if lines[index - 1][-1] != ')' and '=' not in lines[index]:
                lines[index - 1] += lines.pop(index)
            else:
                index += 1

        all_productions = list()

        # Here we have all the rules, line by line in a list
        while True:
            type_block = lines[line_no]
            type_name = type_block[:type_block.find('=')].strip()  # mod, stmt, expr ...
            types.append(type_name)
            constructors_blocks = type_block[type_block.find('=') + 1:].split('|')  # right-line constructor of the type
            i = line_no + 1
            while i < len(lines) and lines[i].strip().startswith('|'):
                t = lines[i].strip()
                cont_constructors_blocks = t[1:].split('|')
                constructors_blocks.extend(cont_constructors_blocks)

                i += 1

            constructors_blocks = list(filter(lambda x: x.strip(), constructors_blocks))  # raw constructors

            # parse type name
            new_type = ASDLPrimitiveType(type_name) if type_name in primitive_type_names else ASDLCompositeType(
                type_name)
            constructors = list(map(_parse_constructor_from_text, constructors_blocks))  # parsed constructors
            fields += [parse_field(constr) for constr in constructors_blocks]
            productions = list(map(lambda c: ASDLProduction(new_type, c), constructors))
            all_productions.extend(productions)

            line_no = i
            if line_no == len(lines):
                break

        # fields = set(fields)
        # field2id = {field: i for i, field in enumerate(fields)}
        # types = set(types)
        # type2id = {type: i for i, type in enumerate(types)}

        grammar = Grammar(all_productions)

        return all_productions, grammar, primitive_type_names


class ASDLProduction(object):
    def __init__(self, type, constructor):
        self.type = type
        self.constructor = constructor

    @property
    def fields(self):
        return self.constructor.fields

    def __getitem__(self, field_name):
        return self.constructor[field_name]

    def __hash__(self):
        h = hash(self.type) ^ hash(self.constructor)

        return h

    def __eq__(self, other):
        return isinstance(other, ASDLProduction) and \
               self.type == other.type and \
               self.constructor == other.constructor

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self):
        return '%s -> %s' % (self.type.__repr__(plain=True), self.constructor.__repr__(plain=True))


class ASDLConstructor(object):
    def __init__(self, name, fields=None):
        self.name = name
        self.fields = []
        if fields:
            self.fields = list(fields)

    def __getitem__(self, field_name):
        for field in self.fields:
            if field.name == field_name: return field

        raise KeyError

    def __hash__(self):
        h = hash(self.name)
        for field in self.fields:
            h ^= hash(field)

        return h

    def __eq__(self, other):
        return isinstance(other, ASDLConstructor) and \
               self.name == other.name and \
               self.fields == other.fields

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self, plain=False):
        plain_repr = '%s(%s)' % (self.name,
                                 ', '.join(f.__repr__(plain=True) for f in self.fields))
        if plain:
            return plain_repr
        else:
            return 'Constructor(%s)' % plain_repr


class Field(object):
    def __init__(self, name, type, cardinality):
        self.name = name
        self.type = type

        assert cardinality in ['-', '?', '*', '+']
        self.cardinality = cardinality

    def __hash__(self):
        h = hash(self.name) ^ hash(self.type)
        h ^= hash(self.cardinality)

        return h

    def __eq__(self, other):
        return isinstance(other, Field) and \
               self.name == other.name and \
               self.type == other.type and \
               self.cardinality == other.cardinality

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self, plain=False):
        plain_repr = '%s%s %s' % (self.type.__repr__(plain=True),
                                  Field.get_cardinality_repr(self.cardinality),
                                  self.name)
        if plain:
            return plain_repr
        else:
            return 'Field(%s)' % plain_repr

    @staticmethod
    def get_cardinality_repr(cardinality):
        return '' if cardinality == 'single' else '?' if cardinality == 'optional' else '*'


class ASDLType(object):
    def __init__(self, type_name):
        self.name = type_name

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return isinstance(other, ASDLType) and self.name == other.name

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self, plain=False):
        plain_repr = self.name
        if plain:
            return plain_repr
        else:
            return '%s(%s)' % (self.__class__.__name__, plain_repr)


class ASDLCompositeType(ASDLType):
    pass


class ASDLPrimitiveType(ASDLType):
    pass


class GrammarRule:
    def __init__(self, label, type, field):  # Type = lhs, field = rhs + rhs_names
        # label must be a python ASDL constructor,rhs_names must be the exact same names as in the ASDL specification.
        self.label = label
        self.type = type
        self.field = field
        self.rhs = []
        self.rhs_names = []
        self.iter_flags = []
        for value in self.field:
            self.rhs.append(value.type.name)
            self.rhs_names.append(value.name)
            self.iter_flags.append(value.cardinality)
        assert (len(self.rhs) == len(self.rhs_names))

    def arity(self):
        return len(self.rhs)

    def is_applicable(self, context):
        # contexts are couples (symbol,iterflag)
        symbol, iterflag = context[0]
        return symbol == self.type

    def apply(self, context, primitives):
        # mods = '+','*','-'
        if self.rhs:
            symbol, iterflag = context[0]
            if iterflag == '+':
                zrhs = list(zip(self.rhs, self.iter_flags))
                context[0] = (symbol, '*')
                return zrhs + context
            elif iterflag == '-' or iterflag == '?':
                zrhs = list(zip(self.rhs, self.iter_flags))
                return zrhs + context[1:]
            elif iterflag == '*':
                zrhs = list(zip(self.rhs, self.iter_flags))
                return zrhs + context
            # elif iterflag == '?':
            #     zrhs = list(zip(self.rhs, self.iter_flags))
            #     return zrhs + context[1:]
        elif self.type not in primitives:
            # print(context)
            if context[0][1] == '-':
                # print('on est en - ')
                return context[1:]
            else:
                # print('on est en + ')
                context[0] = (context[0][0], '*')
                return context
        else:
            return context[1:]

    def build_ast(self, children):
        py_node_type = getattr(sys.modules['ast'], self.label)
        py_ast_node = py_node_type()
        # children is a list of AST objects
        for key, value in zip(self.rhs_names, children):
            setattr(py_ast_node, key, value)
        return py_ast_node

    def __str__(self):
        return self.label


class ReduceAction:
    def __init__(self, label):
        self.label = label
        self.rhs = []

    def is_applicable(self, context):
        symbol, iterflag = context[0]
        return iterflag == '*'

    def apply(self, context, primitives):
        return context[1:]


class ParentInfo:
    def __init__(self, type, field):
        self.type = type
        self.field = field
