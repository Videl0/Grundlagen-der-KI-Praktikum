import time              # für Zeitmessung der Experimente
import random            # für Zufallsauswahl in Min-Conflicts

from csp import (
    Zebra,               # fertiges CSP-Modell des Einstein/Zebra-Rätsels
    backtracking_search, # allgemeiner Backtracking-Algorithmus
    first_unassigned_variable,
    unordered_domain_values,
    no_inference,
    num_legal_values,
    AC3,
)


def degree_heuristic(assignment, csp):
    # wählt aus den noch unbelegten Variablen die mit den meisten Nachbarn (höchster Grad)
    unassigned = [v for v in csp.variables if v not in assignment]
    return max(unassigned, key=lambda var: len(csp.neighbors[var]))


def mrv_plus_degree(assignment, csp):
    # kombiniert MRV (Minimum Remaining Values) + Gradheuristik als Tie-Breaker
    unassigned = [v for v in csp.variables if v not in assignment]

    def domain_size(var):
        if csp.curr_domains:                                  # wenn AC-3 o.ä. schon Domains eingeschränkt hat
            return len(csp.curr_domains[var])                 # nutzt die aktuelle Domänengröße
        return num_legal_values(csp, var, assignment)         # sonst: zählt zulässige Werte aus der Originaldomäne

    sizes = {v: domain_size(v) for v in unassigned}           # Domänengröße je Variable
    min_size = min(sizes.values())                            # kleinste (MRV)
    mrv_candidates = [v for v in unassigned if sizes[v] == min_size]  # alle mit minimaler Restdomäne

    if len(mrv_candidates) == 1:
        return mrv_candidates[0]                              # eindeutige MRV-Variable

    return max(mrv_candidates, key=lambda var: len(csp.neighbors[var]))  # bei Gleichstand: höchste Nachbarschaft (Grad)


def pretty_print_zebra_solution(solution):
    if solution is None:
        print("Keine Lösung gefunden.")
        return
    for house in range(1, 6):  # Häuser sind 1..5
        attrs = [var for (var, val) in solution.items() if val == house]  # alle Eigenschaften, die dieses Haus haben
        print(f"Haus {house}: " + ", ".join(sorted(attrs)))
    print()


def experiment_1_basic_bt():
    csp = Zebra()  # neue CSP-Instanz für das Rätsel
    start = time.perf_counter()
    solution = backtracking_search(
        csp,
        select_unassigned_variable=first_unassigned_variable,  # einfache Reihenfolge (kein MRV)
        order_domain_values=unordered_domain_values,           # Werte in gegebener Reihenfolge
        inference=no_inference                                # keine zusätzliche Inferenz
    )
    end = time.perf_counter()

    print("=== Experiment 1: Basis-Backtracking ===")
    pretty_print_zebra_solution(solution)
    print(f"nassigns: {csp.nassigns}")                        # wie oft eine Variable belegt wurde
    print(f"Laufzeit: {end - start:.4f}s\n")


def experiment_2_mrv_degree():
    csp = Zebra()
    start = time.perf_counter()
    solution = backtracking_search(
        csp,
        select_unassigned_variable=mrv_plus_degree,          # MRV + Gradheuristik
        order_domain_values=unordered_domain_values,
        inference=no_inference
    )
    end = time.perf_counter()

    print("=== Experiment 2: MRV + Gradheuristik ===")
    pretty_print_zebra_solution(solution)
    print(f"nassigns: {csp.nassigns}")
    print(f"Laufzeit: {end - start:.4f}s\n")


def experiment_3_ac3_then_bt():
    csp = Zebra()

    print("=== Experiment 3: AC-3 vor Backtracking ===")
    start_ac3 = time.perf_counter()
    consistent, checks = AC3(csp)                             # Konsistenzprüfung und Domänenreduktion
    end_ac3 = time.perf_counter()

    print(f"AC-3 Konsistent? {consistent}")
    print(f"AC-3 Checks: {checks}")
    print(f"AC-3 Laufzeit: {end_ac3 - start_ac3:.4f}s")

    inferred = csp.infer_assignment()                         # alle durch AC-3 schon bestimmten Variablen
    if consistent and len(inferred) == len(csp.variables):    # wenn alles schon feststeht
        print("AC-3 liefert vollständige Lösung:")
        pretty_print_zebra_solution(inferred)
        print(f"nassigns: {csp.nassigns}")
        print()
        return

    print("AC-3 unvollständig, starte Backtracking mit MRV+Grad...")

    start_bt = time.perf_counter()
    solution = backtracking_search(
        csp,
        select_unassigned_variable=mrv_plus_degree,           # gleiche Heuristik wie in Experiment 2
        order_domain_values=unordered_domain_values,
        inference=no_inference
    )
    end_bt = time.perf_counter()

    pretty_print_zebra_solution(solution)
    print(f"nassigns: {csp.nassigns}")
    print(f"Backtracking-Laufzeit: {end_bt - start_bt:.4f}s\n")


def min_conflicts_local(csp, max_steps=20000):
    # startete mit einer zufälligen vollständigen Belegung
    current = {
        var: random.choice(list(csp.domains[var]))            # für jede Variable irgendein Wert aus der Domäne
        for var in csp.variables
    }

    for _ in range(max_steps):
        conflicted = [
            v for v in csp.variables
            if csp.nconflicts(v, current[v], current) > 0     # Variablen die aktuell an mindestens einem Konflikt beteiligt sind
        ]
        if not conflicted:
            return current                                    # keine Konflikte mehr → Lösung gefunden

        var = random.choice(conflicted)                       # wählt zufällige konfliktbehaftete Variable
        domain = csp.domains[var]
        best_val = min(
            domain,
            key=lambda val: csp.nconflicts(var, val, current) # wählt Wert mit minimalen Konflikten
        )
        current[var] = best_val                               # setzt Variable auf konfliktärmsten Wert

    return None                                               # keine Lösung innerhalb der Schrittbegrenzung gefunden


def experiment_4_min_conflicts(max_steps=20000):
    csp = Zebra()

    start = time.perf_counter()
    solution = min_conflicts_local(csp, max_steps=max_steps)
    end = time.perf_counter()

    print("=== Experiment 4: Min-Conflicts (lokal) ===")
    pretty_print_zebra_solution(solution)
    print(f"Laufzeit: {end - start:.4f}s\n")


if __name__ == "__main__":
    experiment_1_basic_bt()
    experiment_2_mrv_degree()
    experiment_3_ac3_then_bt()
    experiment_4_min_conflicts()
