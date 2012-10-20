ranked_alternatives_mf20 = {}
for k, v in altset.iteritems():
    ranked = []
    for verb in v:
        if "_" in verb:
            pass
        else:
            score = getSentenceScore(LM, verb)
            ranked.append((verb, score))
    ranked = sorted(ranked, key=lambda x: x[1], reverse=True)
    ranked_alternatives_mf20[k] = ranked[:20]


