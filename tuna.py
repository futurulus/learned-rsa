#!/usr/bin/env python

import re
from glob import iglob
import xml.etree.ElementTree as ET
from tokenizers import basic_unigram_tokenizer


class TunaCorpus:
    def __init__(self, filenames):
        self.filenames = filenames

    def iter_trials(self):
        for filename in self.filenames:
            yield Trial(filename)


class Trial:
    def __init__(self, filename):
        self.filename = filename
        tree = ET.parse(self.filename)
        root = tree.getroot()
        # General trial-level attributes: cardinality, condition, domain, id
        for key, val in root.attrib.items():
            try:
                val = int(val)
            except:
                pass
            setattr(self, key.lower(), val)
        # The set of entities in the <DOMAIN> element (there is always exactly 1):
        self.entities = [Entity(e) for e in root[0].iter('ENTITY')]
        self.targets = [e for e in self.entities if e.is_target()]
        # <STRING-DESCRIPTION> is always unique and the the 2nd daughter of the root:
        string_description_elem = root[1]
        # <DESCRIPTION> is always unique and the the 3rd daughter of the root:
        description_elem = root[2]
        # <ATTRIBUTE-SET> is always unique and the the 4th/final daughter of the root:
        attribute_set_elem = root[3]
        # More work needs to be done on descriptions if we want to use them beyond
        # string_description. For now, only string_description is fully configured.
        self.description = Description(string_description_elem, description_elem, attribute_set_elem)


class Entity:
    def __init__(self, element):
         # General entity-level attributes: id, image, type
        for key, val in element.attrib.items():
            try:
                val = int(val)
            except:
                pass
            setattr(self, key.lower(), val)
        self.attributes = [Attribute(e) for e in element.iter('ATTRIBUTE')]

    def is_target(self):
        return self.type == "target"

    def attributes_as_dict(self):
        return {a.name: a.value for a in self.attributes}

    def __eq__(self, e):
        """Defines equality in terms of equality of attributes"""
        if len(self.attributes) != len(e.attributes):
            return False
        return not False in [aself == a for aself, a in zip(self.attributes, e.attributes)]

    def __ne__(self, e):
        return not self.__eq__(e)

    def __contains__(self, a):
        for x in self.attributes:
            if x == a:
                return True
        return False


class Description:
    def __init__(self, string_description_elem, description_elem, attribute_set_elem):
        self.string_description = re.sub(r"\s*\n\s*", " ", string_description_elem.text.strip())
        self.description = description_elem
        self.attribute_set = [Attribute(a) for a in attribute_set_elem.iter('ATTRIBUTE')]

    def unigrams(self):
        return basic_unigram_tokenizer(self.string_description)


class Attribute:
    def __init__(self, element):
        self.type = None
        for key, val in element.attrib.items():
            setattr(self, key.lower(), val)

    def __str__(self):
        return ":".join([x for x in [self.name, self.value] if x])

    def __eq__(self, a):
        return (self.type==a.type) and (self.name==a.name) and (self.value==a.value)

    def __ne__(self, a):
        return not self.__eq__(a)

    def __hash__(self):
        return hash(str(self))

    def __lt__(self, a):
        return str(self) < str(a)


if __name__ == '__main__':

    from collections import Counter
    from operator import itemgetter

    all_filenames = iglob("../TUNA/corpus/*/*/*.xml")


    corpus = TunaCorpus(all_filenames)
    counts = Counter([w for t in corpus.iter_trials() for w in t.description.unigrams()])
    for key, val in sorted(counts.items(), key=itemgetter(1), reverse=False):
        print key, val
    print
    print 'Vocab size:', len(counts)
    print 'Tokens:', sum(counts.values())
