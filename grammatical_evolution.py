from pyneurgen.grammatical_evolution import GrammaticalEvolution
from pyneurgen.fitness import FitnessElites, FitnessTournament
from pyneurgen.fitness import ReplacementTournament, MAX, MIN, CENTER

bnf = """
<expr>              ::= <expr> <biop> <expr> | <uop> <expr> | <real> |
                        math.log(abs(<expr>)) | <pow> | math.sin(<expr> )|
                        value | (<expr>)
<biop>              ::= + | - | * | /
<uop>               ::= + | -
<pow>               ::= pow(<expr>, <real>)
<plus>              ::= +
<minus>             ::= -
<real>              ::= <int-const>.<int-const>
<int-const>         ::= <int-const> | 1 | 2 | 3 | 4 | 5 | 6 |
                        7 | 8 | 9 | 0
<S>                 ::=
import math
total = 0.0
for i in xrange(100):
    value = float(i) / float(100)
    total += abs(<expr> - pow(value, 3))
fitness = total
self.set_bnf_variable('<fitness>', fitness)
        """

ges = GrammaticalEvolution()

ges.set_bnf(bnf)
ges.set_genotype_length(start_gene_length=20,
                        max_gene_length=500)
ges.set_population_size(50)
ges.set_wrap(True)

ges.set_max_generations(1000)
ges.set_fitness_type(MIN, target_value=0.01)

ges.set_max_program_length(500)
ges.set_timeouts(10, 120)
ges.set_fitness_fail(100.0)


ges.set_fitness_selections(
    FitnessElites(ges.fitness_list, .05),
    FitnessTournament(ges.fitness_list, tournament_size=2))
ges.set_max_fitness_rate(.5)


ges.set_mutation_rate(.025)
ges.set_fitness_selections(
    FitnessElites(ges.fitness_list, .05),
    FitnessTournament(ges.fitness_list, tournament_size=2))
ges.set_max_fitness_rate(.5)

ges.set_crossover_rate(.2)
ges.set_children_per_crossover(2)
ges.set_mutation_type('m')
ges.set_max_fitness_rate(.25)

ges.set_replacement_selections(
    ReplacementTournament(ges.fitness_list, tournament_size=3))

ges.set_maintain_history(True)

ges.create_genotypes()
print ges.run()
