# Astrozyten

Astrozyten sind die häufigsten Gliazellen im Zentralnervensystem. Sie regulieren
das extrazelluläre Kalium, nehmen Glutamat aus dem Synapsenspalt auf und sind an
der Blut-Hirn-Schranke beteiligt. Über Calcium-Signale und die Freisetzung von
Gliotransmittern wie ATP, D-Serin und Glutamat können sie die synaptische
Übertragung modulieren. Man spricht deshalb von der tripartiten Synapse aus
präsynaptischer Endigung, postsynaptischer Membran und Astrozytenfortsatz.

# Izhikevich-Modell

Das 2003 von Eugene Izhikevich vorgeschlagene Neuronmodell beschreibt die
Membrandynamik mit zwei gekoppelten Differentialgleichungen:
v' = 0.04 v^2 + 5 v + 140 - u + I und u' = a (b v - u).
Erreicht v einen Schwellwert von 30 mV, wird v auf c gesetzt und u um d erhöht.
Mit vier Parametern (a, b, c, d) reproduziert es über zwanzig Feuermuster
kortikaler Neurone bei einem Bruchteil der Rechenkosten des Hodgkin-Huxley-Modells.

# Calcium-Zeitskalen in ASTRA_Glia

ASTRA_Glia nutzt drei getrennte Calcium-Kanäle mit unterschiedlichen Zeitkonstanten.
Der schnelle Kanal (tau etwa 150 ms) wirkt pro Domäne als exzitatorischer Gain.
Der mittlere Kanal (tau etwa 5 s) wirkt pro Neuron dämpfend und hält die
Feuerrate im Zielband. Der langsame Kanal (tau etwa 10 s) ist global und dient
als Tor für Konsolidierung: nur wenn genug Salienz aufgelaufen ist, wird eine
Erinnerung vom Kurzzeit- ins Langzeitgedächtnis überführt.

# Homöostatische Plastizität

Homöostatische Plastizität stabilisiert die Aktivität neuronaler Netze. Bei
dauerhaft zu hoher oder zu niedriger Feuerrate werden synaptische Gewichte
global skaliert (synaptic scaling, beschrieben von Gina Turrigiano) oder die
intrinsische Erregbarkeit angepasst. Im Gegensatz zur schnellen Hebb'schen
Plastizität wirkt dieser Regelkreis über Stunden bis Tage und verhindert,
dass sich Lernprozesse selbst verstärken und das Netz destabilisieren.

# Drei-Faktor-Lernregel

Reine Hebb-Regeln kennen nur Prä- und Postsynapsenaktivität. Drei-Faktor-Regeln
ergänzen einen dritten, meist neuromodulatorischen Faktor wie Dopamin oder
Acetylcholin. Prä- und Postaktivität erzeugen zunächst einen stillen
Eligibility Trace, eine synaptische Markierung mit exponentiellem Zerfall.
Erst der dritte Faktor wandelt diese Markierung in eine dauerhafte
Gewichtsänderung um. So lässt sich eine Belohnung auch dann noch zuordnen,
wenn sie Sekunden nach der eigentlichen Aktivität eintrifft.

# Adulte Neurogenesis

Beim erwachsenen Säuger werden neue Neurone vor allem in der Subgranularzone
des Dentate Gyrus im Hippocampus und in der subventrikulären Zone gebildet.
Ob adulte Neurogenesis beim Menschen in nennenswertem Umfang stattfindet,
ist umstritten: Studien von Sorrells und Kollegen (2018) fanden kaum neue
Neurone, während Boldrini und Moreno-Jiménez im selben Jahr bzw. 2019
deren Persistenz bis ins hohe Alter berichteten.

# Quellen

- Izhikevich, E. M. (2003): Simple Model of Spiking Neurons. IEEE Trans. Neural Networks 14(6).
- Turrigiano, G. G. et al. (1998): Activity-dependent scaling of quantal amplitude in neocortical neurons. Nature 391.
- Sorrells, S. F. et al. (2018): Human hippocampal neurogenesis drops sharply in children to undetectable levels in adults. Nature 555.
- Boldrini, M. et al. (2018): Human hippocampal neurogenesis persists throughout aging. Cell Stem Cell 22.
- Moreno-Jiménez, E. P. et al. (2019): Adult hippocampal neurogenesis is abundant in neurologically healthy subjects and drops sharply in patients with Alzheimer's disease. Nature Medicine 25.