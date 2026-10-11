**CaP-X plugin:** Request Python-only replies and comment a limited leading prose
prefix when all remaining Python compiles, avoiding an extra model call for
recoverable formatting errors. Original replies, normalized code, and a brief
notice remain in transcripts. Policy configs record the code-format version
for comparisons across assistance profiles. Recovery is a lexical heuristic;
it does not validate the manipulation plan.
