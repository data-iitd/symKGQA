from datetime import date

def comp(a, b, op):
    """
    Args:
        - a (ValueClass): attribute value of a certain entity
        - b (ValueClass): comparison target
        - op: =/>/</!=
    Example:
        a is someone's birthday, 1960-02-01, b is 1960, op is '=', then return True
    """

    if b.isTime():
        # Note: for time, 'a=b' actually means a in b, 'a!=b' means a not in b
        if op == '=':
            return b.contains(a)
        elif op == '!=':
            return not b.contains(a)
    if op == '=':
        return a == b
    elif op == '<':
        return a < b
    elif op == '>':
        return a > b
    elif op == '!=':
        return a != b
    
def isOp(op):
    if (op in ['=', '!=', '<', '>']):
        return True
    else:
        return False

class ValueClass():
    def __init__(self, type, value, unit=None):
        """
        When type is
            - string, value is a str
            - quantity, value is a number and unit is required
            - year, value is a int
            - date, value is a date object
        """
        self.type = type
        self.value = value
        self.unit = unit
        self.preDefinedUnits = {'minute': ['minutes', 'minute'], 'metre': ['metres', 'metre']}

    def isTime(self):
        return self.type in {'year', 'date'}

    def can_compare(self, other):
        if self.type == 'string':
            if(other.type == 'string'):
                return True
            elif(other.type == 'quantity'):
                try:
                    other.value = str(int(other.value))
                    other.type = 'string'
                    print("Changing datatype to string!")
                except:
                    pass
            else:
                try:
                    other.value = str(other.value)
                    other.type = 'string'
                    print("Changing datatype to string!")
                except:
                    pass

            return other.type == 'string'
        elif self.type == 'quantity':
            # NOTE: for two quantity, they can compare only when they have the same unit
            if(other.type == 'string' or other.type == 'year'):
                other.value = str(other.value)
                try:
                    if ' ' in other.value:
                        vs = other.value.split()
                        v = vs[0]
                        unit = ' '.join(vs[1:])
                        other.value = float(v)
                        other.unit = unit
                    else:
                        v = other.value
                        unit = '1'
                        other.value = float(v)
                        other.unit = unit
                    print("Changing datatype to quantity!")
                except:
                    pass
            

            if(self.unit in self.preDefinedUnits.keys()):
                return other.type == 'quantity' and other.unit in self.preDefinedUnits[self.unit]
            elif(other.unit in self.preDefinedUnits.keys()):
                return other.type == 'quantity' and self.unit in self.preDefinedUnits[other.unit]
            return other.type == 'quantity' and other.unit == self.unit
        else:
            # year can compare with date
            if(other.type == 'quantity' or other.type == 'string'):
                try:
                    value = other.value
                    if(other.type == 'quantity'):
                        value = str(int(other.value))

                    if '/' in value or ('-' in value and '-' != value[0]):
                        split_char = '/' if '/' in value else '-'
                        p1, p2 = value.find(split_char), value.rfind(split_char)
                        y, m, d = int(value[:p1]), int(value[p1+1:p2]), int(value[p2+1:])
                        other.value = date(y, m, d)
                        other.type = 'date'
                    else:
                        other.value = int(value)
                        other.type = 'year'
                    print("Changing datatype to date/year!")
                except:
                    pass

            


            return other.type == 'year' or other.type == 'date'

    def contains(self, other):
        """
        check whether self contains other, which is different from __eq__ and the result is asymmetric
        used for conditions like whether 2001-01-01 in 2001, or whether 2001 in 2001-01-01
        """
        if self.type == 'year': # year can contain year and date
            other_value = other.value if other.type == 'year' else other.value.year
            return self.value == other_value
        elif self.type == 'date': # date can only contain date
            return other.type == 'date' and self.value == other.value
        else:
            raise Exception('not supported type: %s' % self.type)


    def __eq__(self, other):
        """
        2001 and 2001-01-01 is not equal
        """
        assert self.can_compare(other)
        return self.type == other.type and self.value == other.value

    def __lt__(self, other):
        """
        Comparison between a year and a date will convert them both to year
        """
        assert self.can_compare(other)
        if self.type == 'string':
            raise Exception('try to compare two string')
        elif self.type == 'quantity':
            return self.value < other.value
        elif self.type == 'year':
            other_value = other.value if other.type == 'year' else other.value.year
            return self.value < other_value
        elif self.type == 'date':
            if other.type == 'year':
                return self.value.year < other.value
            else:
                return self.value < other.value

    def __gt__(self, other):
        assert self.can_compare(other)
        if self.type == 'string':
            raise Exception('try to compare two string')
        elif self.type == 'quantity':
            return self.value > other.value
        elif self.type == 'year':
            other_value = other.value if other.type == 'year' else other.value.year
            return self.value > other_value
        elif self.type == 'date':
            if other.type == 'year':
                return self.value.year > other.value
            else:
                return self.value > other.value

    def __str__(self):
        if self.type == 'string':
            return self.value
        elif self.type == 'quantity':
            if self.value - int(self.value) < 1e-5:
                v = int(self.value)
            else:
                v = self.value
            return '{} {}'.format(v, self.unit) if self.unit != '1' else str(v)
        elif self.type == 'year':
            return str(self.value)
        elif self.type == 'date':
            return self.value.isoformat()
