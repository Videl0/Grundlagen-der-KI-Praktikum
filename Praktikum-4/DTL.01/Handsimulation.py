# tree2.py
# ID3 und CAL3 Implementierung mit Baumknoten-Klassen

import math
import random
from collections import defaultdict

# ------------------------- Hilfen ---------------------------

def remove_all(item, seq):
    # Entfernt alle Vorkommen von item aus seq
    return [x for x in seq if x != item]

def normalize(numbers):
    # Normiert eine Liste auf Summe 1
    s = sum(numbers)
    if s == 0:
        return numbers
    return [n / s for n in numbers]

def argmax_random_tie(seq, key):
    # Wählt ein max-Element nach key, bei Gleichstand zufällig
    items = list(seq)
    random.shuffle(items)
    return max(items, key=key)

def information_content(values):
    # Entropie in Bit für Häufigkeiten
    probs = normalize([v for v in values if v != 0])
    return sum(-p * math.log2(p) for p in probs)

# ------------------------- Daten ----------------------------

class DataSet:
    # Datenstruktur wie in AIMA
    def __init__(self, examples, values, attr_names, target, inputs):
        self.examples = examples
        self.values = values
        self.attr_names = attr_names
        self.target = target
        self.inputs = inputs

# ----------------------- Baumknoten -------------------------

class DecisionFork:
    # Testknoten mit Attribut und Kindern je Attributwert
    def __init__(self, attr, attr_name=None, default_child=None, branches=None):
        self.attr = attr
        self.attr_name = attr_name or str(attr)
        self.default_child = default_child
        self.branches = branches or {}

    def add(self, val, subtree):
        # Fügt einen Zweig hinzu
        self.branches[val] = subtree

    def __call__(self, example):
        # Klassifiziert ein Beispiel
        val = example[self.attr]
        if val in self.branches:
            return self.branches[val](example)
        return self.default_child(example)

class DecisionLeaf:
    # Blatt mit Ergebnislabel
    def __init__(self, result):
        self.result = result

    def __call__(self, example):
        return self.result

# ------------------------- ID3 ------------------------------

def ID3Learner(dataset):
    # Lernfunktion für ID3
    target, values = dataset.target, dataset.values

    def plurality_value(examples):
        # Blatt mit Mehrheitsklasse
        return DecisionLeaf(argmax_random_tie(values[target], key=lambda v: sum(e[target] == v for e in examples)))

    def all_same_class(examples):
        # True wenn alle gleiche Klasse
        if not examples:
            return True
        c0 = examples[0][target]
        return all(e[target] == c0 for e in examples)

    def split_by(attr, examples):
        # Teilt nach Attributwerten
        return [(v, [e for e in examples if e[attr] == v]) for v in values[attr]]

    def information_gain(attr, examples):
        # Informationsgewinn für Attribut
        def I(exs):
            counts = [sum(e[target] == v for e in exs) for v in values[target]]
            return information_content(counts)
        n = len(examples)
        parts = split_by(attr, examples)
        remainder = sum((len(exs) / n) * I(exs) for (v, exs) in parts if len(exs) > 0)
        return I(examples) - remainder

    def choose_attribute(attrs, examples):
        # Wählt Attribut mit maximalem Informationsgewinn
        return argmax_random_tie(attrs, key=lambda a: information_gain(a, examples))

    def dtl(examples, attrs, parent_examples):
        # Rekursive Baumkonstruktion
        if len(examples) == 0:
            return plurality_value(parent_examples)
        if all_same_class(examples):
            return DecisionLeaf(examples[0][target])
        if len(attrs) == 0:
            return plurality_value(examples)
        A = choose_attribute(attrs, examples)
        tree = DecisionFork(A, dataset.attr_names[A], plurality_value(examples))
        for (v_k, exs) in split_by(A, examples):
            subtree = dtl(exs, remove_all(A, attrs), examples)
            tree.add(v_k, subtree)
        return tree

    return dtl(dataset.examples, dataset.inputs, ())

# ------------------------- CAL3 -----------------------------

def CAL3Learner(dataset, S1=4, S2=0.7):
    # Lernfunktion für CAL3 mit Stopps S1 und S2
    target, values = dataset.target, dataset.values

    def plurality_value(examples):
        # Blatt mit Mehrheitsklasse
        return DecisionLeaf(argmax_random_tie(values[target], key=lambda v: sum(e[target] == v for e in examples)))

    def purity(examples):
        # Anteil der häufigsten Klasse
        if not examples:
            return 0.0
        counts = [sum(e[target] == v for e in examples) for v in values[target]]
        return max(counts) / float(len(examples))

    def all_same_class(examples):
        # True wenn alle gleiche Klasse
        if not examples:
            return True
        c0 = examples[0][target]
        return all(e[target] == c0 for e in examples)

    def split_by(attr, examples):
        # Teilt nach Attributwerten
        return [(v, [e for e in examples if e[attr] == v]) for v in values[attr]]

    def information_gain(attr, examples):
        # Informationsgewinn für Attribut
        def I(exs):
            counts = [sum(e[target] == v for e in exs) for v in values[target]]
            return information_content(counts)
        n = len(examples)
        parts = split_by(attr, examples)
        remainder = sum((len(exs) / n) * I(exs) for (v, exs) in parts if len(exs) > 0)
        return I(examples) - remainder

    def choose_attribute(attrs, examples):
        # Wählt Attribut mit maximalem Informationsgewinn
        return argmax_random_tie(attrs, key=lambda a: information_gain(a, examples))

    def cal3(examples, attrs, parent_examples):
        # Rekursive Baumkonstruktion mit Stopps
        if len(examples) == 0:
            return plurality_value(parent_examples)
        if all_same_class(examples):
            return DecisionLeaf(examples[0][target])
        if len(attrs) == 0:
            return plurality_value(examples)
        if len(examples) < S1 or purity(examples) >= S2:
            return plurality_value(examples)
        A = choose_attribute(attrs, examples)
        tree = DecisionFork(A, dataset.attr_names[A], plurality_value(examples))
        for (v_k, exs) in split_by(A, examples):
            subtree = cal3(exs, remove_all(A, attrs), examples)
            tree.add(v_k, subtree)
        return tree

    return cal3(dataset.examples, dataset.inputs, ())

# ----------------------- Pretty-Print (Für schönere Ausgabe) ------------------------

def pretty(tree, dataset, indent=""):
    # Ausgabe eines Baumes mit Attributnamen aus dataset
    if isinstance(tree, DecisionLeaf):
        return indent + "-> " + str(tree.result)
    # DecisionFork Fall
    s = indent + "[" + dataset.attr_names[tree.attr] + "]\n"
    # Werte in derselben Reihenfolge wie dataset.values
    for val in dataset.values[tree.attr]:
        if val in tree.branches:
            s += indent + "  " + dataset.attr_names[tree.attr] + " = " + str(val) + ":\n"
            s += pretty(tree.branches[val], dataset, indent + "    ") + "\n"
    return s.rstrip()

# -------------------------- Demo ----------------------------

if __name__ == "__main__":
    # Beispiele als Listen in der Reihenfolge [Alter, Einkommen, Bildung, Kandidat]
    examples = [
        ['>=35', 'hoch', 'Abitur',   'O'],
        ['<35',  'niedrig', 'Master',   'O'],
        ['>=35', 'hoch',    'Bachelor', 'M'],
        ['>=35', 'niedrig', 'Abitur',   'M'],
        ['>=35', 'hoch',    'Master',   'O'],
        ['<35',  'hoch',    'Bachelor', 'O'],
        ['<35',  'niedrig', 'Abitur',   'M'],
    ]
    # Mögliche Werte pro Attribut (in Ausgabe-Reihenfolge)
    values = [
        ['>=35', '<35'],                  # Alter
        ['hoch', 'niedrig'],              # Einkommen
        ['Abitur', 'Bachelor', 'Master'], # Bildung
        ['O', 'M']                        # Kandidat
    ]
    # Attributnamen für Ausgabe
    attr_names = ['Alter', 'Einkommen', 'Bildung', 'Kandidat']
    # DataSet anlegen
    ds = DataSet(examples, values, attr_names, target=3, inputs=[0, 1, 2])

    # ID3 lernen und schön ausgeben
    id3_tree = ID3Learner(ds)
    print("ID3 Baum:\n")
    print(pretty(id3_tree, ds))

    # CAL3 lernen und schön ausgeben
    cal3_tree = CAL3Learner(ds, S1=4, S2=0.7)
    print("\nCAL3 Baum (S1=4, S2=0.7):\n")
    print(pretty(cal3_tree, ds))

    # Beispielvorhersagen im gleichen Stil wie oben
    def predict_with_dict(tree, d):
        # Wandelt Dict in Beispiel-Liste um
        ex = [d['Alter'], d['Einkommen'], d['Bildung'], None]
        return tree(ex)

    print("\nVorhersagen mit ID3:")
    q1 = {'Alter': '>=35', 'Einkommen': 'hoch',    'Bildung': 'Abitur'}
    q2 = {'Alter': '<35',  'Einkommen': 'niedrig', 'Bildung': 'Abitur'}
    q3 = {'Alter': '<35',  'Einkommen': 'hoch',    'Bildung': 'Bachelor'}
    print(q1, "->", predict_with_dict(id3_tree, q1))
    print(q2, "->", predict_with_dict(id3_tree, q2))
    print(q3, "->", predict_with_dict(id3_tree, q3))

    print("\nVorhersagen mit CAL3:")
    print(q1, "->", predict_with_dict(cal3_tree, q1))
    print(q2, "->", predict_with_dict(cal3_tree, q2))
    print(q3, "->", predict_with_dict(cal3_tree, q3))

