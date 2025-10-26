import java.util.*; // Utilities für Listen, Zufall, Map
import java.util.function.IntPredicate; // Predikat für Zieltest
import java.util.function.ToIntFunction; // Kostenfunktion

public class GAAllInOne {

    static GAResult runGA(int length, int alphabetSize, int popSize, double pc, double pm, int k,
                          int maxGen, Random rng, ToIntFunction<int[]> costFn, IntPredicate isGoal,
                          int[] seedOrNull) {
        List<int[]> pop = new ArrayList<>(popSize); // Population anlegen
        if (seedOrNull != null) pop.add(seedOrNull.clone()); // optionales Seed einfügen
        while (pop.size() < popSize) pop.add(randInd(length, alphabetSize, rng)); // auffüllen
        int[] best = null;
        int bestCost = Integer.MAX_VALUE; // bestes Individuum tracken

        for (int gen = 1; gen <= maxGen; gen++) { // Generationen
            int[] costs = new int[pop.size()]; // Kostenfeld
            for (int i = 0; i < pop.size(); i++) { // alle Individuen bewerten
                int c = costFn.applyAsInt(pop.get(i)); // Kosten berechnen
                costs[i] = c;
                if (c < bestCost) {
                    bestCost = c;
                    best = pop.get(i).clone();
                } // Bestes merken
            }
            if (isGoal.test(bestCost)) return new GAResult(true, gen, best, bestCost); // Ziel erreicht

            List<int[]> mating = new ArrayList<>(popSize); // Matingpool
            while (mating.size() < popSize) mating.add(tournament(pop, costs, k, rng)); // Turnier Selektion

            List<int[]> next = new ArrayList<>(popSize); // neue Population
            if (best != null) { //NPE
                next.add(best.clone()); // Elitismus
            }
            while (next.size() < popSize) { // Kinder erzeugen
                int[] p1 = mating.get(rng.nextInt(mating.size())); // Eltern 1
                int[] p2 = mating.get(rng.nextInt(mating.size())); // Eltern 2
                int[] child = onePoint(p1, p2, pc, rng); // Einpunkt Crossover
                mutate(child, pm, alphabetSize, rng); // Mutation
                next.add(child); // Kind speichern
            }
            pop = next; // Generation ersetzen
        }
        return new GAResult(false, maxGen, best, bestCost); // kein Erfolg nach Limit
    }

    static int[] randInd(int len, int alpha, Random rng) {
        int[] g = new int[len];
        for (int i = 0; i < len; i++) g[i] = rng.nextInt(alpha);
        return g;
    } // zufälliges Individuum

    static int[] tournament(List<int[]> pop, int[] costs, int k, Random rng) {
        int best = rng.nextInt(pop.size()); // Startkandidat
        for (int i = 1; i < k; i++) {
            int idx = rng.nextInt(pop.size());
            if (costs[idx] < costs[best]) best = idx;
        } // bestes im Turnier
        return pop.get(best).clone(); // Kopie zurück
    }

    static int[] onePoint(int[] a, int[] b, double pc, Random rng) {
        if (rng.nextDouble() >= pc) return a.clone(); // kein Crossover
        int cut = 1 + rng.nextInt(a.length - 1); // Schnittpunkt wählen
        int[] c = new int[a.length]; // Kind anlegen
        for (int i = 0; i < cut; i++) c[i] = a[i]; // Kopf von a (alternative mit System.arraycopy())
        for (int i = cut; i < a.length; i++) c[i] = b[i]; // Rest von b
        return c; // Kind zurück
    }

    static void mutate(int[] g, double pm, int alpha, Random rng) {
        for (int i = 0; i < g.length; i++)
            if (rng.nextDouble() < pm) {
                int o = g[i], n = o;
                if (alpha > 1) while (n == o) n = rng.nextInt(alpha);
                g[i] = n;
            } // neue zufällige Allele
    }

    static int queensConflicts(int[] board) {
        int n = board.length, c = 0; // Konfliktzähler
        for (int i = 0; i < n; i++)
            for (int j = i + 1; j < n; j++) { // alle Paare prüfen
                if (board[i] == board[j]) c++; // gleiche Zeile
                if (Math.abs(board[i] - board[j]) == Math.abs(i - j)) c++; // gleiche Diagonale
            }
        return c; // 0 ist Lösung
    }

    static void runQueens() {
        final int N = 8, runs = 100, pop = 50, k = 3, maxGen = 300;
        final double pc = 0.7, pm = 0.01; // Settings
        int succ = 0, genSum = 0;
        Random master = new Random(); // Statistik und Master RNG
        for (int r = 0; r < runs; r++) {
            Random rng = new Random(master.nextLong()); // pro Lauf eigener RNG
            GAResult rr = runGA(N, N, pop, pc, pm, k, maxGen, rng, GAAllInOne::queensConflicts, cost -> cost == 0, null); // GA starten
            if (rr.success) {
                succ++;
                genSum += rr.generations;
            } // Erfolgszähler
        }
        double sr = succ / (double) runs, aes = succ > 0 ? genSum / (double) succ : Double.NaN; // Metriken
        System.out.printf(Locale.US, "8-Queens  runs=%d  SR=%.2f  AES=%.2f%n", runs, sr, aes); // Ausgabe
    }

    static Graph demoGraph() {
        int n = 12;
        int[][] e = {{0, 1}, {0, 2}, {1, 2}, {1, 3}, {2, 4}, {3, 5}, {4, 5}, {4, 6}, {5, 7}, {6, 7}, {6, 8}, {7, 9}, {8, 9}, {8, 10}, {9, 11}, {10, 11}}; // kleines Beispiel
        return new Graph(n, e); // Graph erstellen
    }

    static int colorConf(Graph G, int[] col) {
        int c = 0;
        for (int u = 0; u < G.n; u++) for (int v : G.adj.get(u)) if (v > u && col[u] == col[v]) c++;
        return c;
    } // Konflikte zählen

    static int usedColors(int[] col, int alpha) {
        boolean[] u = new boolean[alpha];
        int c = 0;
        for (int x : col)
            if (!u[x]) {
                u[x] = true;
                c++;
            }
        return c;
    } // genutzte Farben

    static int colorCost(Graph G, int[] col, int alpha) {
        return 100 * colorConf(G, col) + usedColors(col, alpha);
    } // harte Strafe für Konflikte

    static int[] remapColors(int[] col, int toAlpha) {
        int[] out = col.clone();
        for (int i = 0; i < out.length; i++) out[i] = out[i] % toAlpha;
        return out;
    } // Farben ummappen

    static void runColoring() {
        final Graph G = demoGraph(); // Demo Graph
        final int runs = 100, pop = 120, k = 3, maxGen1 = 600, maxGen2 = 400;
        final double pc = 0.7, pm = 0.02; // nötige Settings (könnte man auch ohne variable machen)
        int succ = 0, genSum = 0;
        Map<Integer, Integer> hist = new TreeMap<>();
        Random master = new Random(); // Statistik

        for (int r = 0; r < runs; r++) {
            Random rng = new Random(master.nextLong()); // pro Lauf RNG
            ToIntFunction<int[]> cost5 = g -> colorCost(G, g, 5); // Kosten mit 5 Farben
            GAResult phase1 = runGA(G.n, 5, pop, pc, pm, k, maxGen1, rng, cost5, c -> c / 100 == 0, null); // erst konfliktfrei mit 5
            if (!phase1.success) continue; // Lauf verworfen wenn nicht konfliktfrei
            succ++;
            genSum += phase1.generations; // Statistik

            int[] cur = phase1.best.clone();
            int min = usedColors(cur, 5); // aktuelle Lösung
            for (int t = Math.min(4, min - 1); t >= 2; t--) { // Farbenzahl testen 4 dann 3 dann 2
                int[] seed = remapColors(cur, t); // Seed auf Zielalphabet
                int finalT = t;
                ToIntFunction<int[]> costT = g -> colorCost(G, g, finalT); // Kosten für t Farben
                GAResult ph = runGA(G.n, t, pop, pc, pm, k, maxGen2, rng, costT, c -> c / 100 == 0, seed); // neuer Versuch
                if (ph.success) {
                    cur = ph.best.clone();
                    min = t;
                    genSum += ph.generations;
                } else break; // übernehmen oder abbrechen
            }
            hist.put(min, hist.getOrDefault(min, 0) + 1); // Histogramm
        }

        double sr = succ / (double) runs, aes = succ > 0 ? genSum / (double) succ : Double.NaN; // Kennzahlen
        System.out.printf(Locale.US, "MapColoring  runs=%d  SR=%.2f  AES=%.2f%n", runs, sr, aes); // Ausgabe
        if (!hist.isEmpty()) {
            System.out.print("Minimale Farben: ");
            boolean f = true; // Verteilung drucken
            for (var e : hist.entrySet()) {
                if (!f) System.out.print(", ");
                System.out.print(e.getKey() + ": " + e.getValue());
                f = false;
            }
            System.out.println();
        }
    }

    public static void main(String[] args) {
        System.out.println("=== GA Experimente ==="); // Startinfo
        long t0 = System.currentTimeMillis(); // Zeitstart
        runQueens(); // 8 Queens ausführen
        runColoring(); // Kartenfärbung ausführen
        long t1 = System.currentTimeMillis(); // Zeitende
        System.out.printf(Locale.US, "Zeit: %.2f s%n", (t1 - t0) / 1000.0); // Laufzeit
    }

    static class GAResult {
        final boolean success;
        final int generations;
        final int[] best;
        final int bestCost; // kompaktes Ergebnis

        GAResult(boolean s, int g, int[] b, int c) {
            success = s;
            generations = g;
            best = b;
            bestCost = c;
        } // Konstruktor
    }

    static class Graph {
        final int n;
        final List<List<Integer>> adj; // Knotenanzahl und Adjazenz

        Graph(int n, int[][] edges) {
            this.n = n;
            adj = new ArrayList<>();
            for (int i = 0; i < n; i++) adj.add(new ArrayList<>()); // Listen anlegen
            for (int[] e : edges) {
                adj.get(e[0]).add(e[1]);
                adj.get(e[1]).add(e[0]);
            }
        } // ungerichtete Kanten
    }
}