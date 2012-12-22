def gen_WNCS(CS):
    CS_WN = collections.OrderedDict()
    voc = CS.keys()
    for v in voc:
        _ag = alt.AlternativeGenerator(v, maxnum=100, score=False, include_hyponyms=True, include_uncertain=True, pos="VB")
        ls = _ag.generate_from_wordnet()
        ls = [_v for _v in ls if _v[0] in voc]
        if ls:
            CS_WN[v] = ls
        else:
            pass
    return CS_WN
