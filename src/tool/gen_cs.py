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

def combineCS(big, small):
    out = collections.OrderedDict()
    for v, cs in big.iteritems():
        _tb = collections.Counter({unicode(t[0]):t[1] for t in cs})
        _ts = {unicode(t[0]):t[1] for t in small[v]} if v in small else {}
        _tb.update(_ts)
        out[unicode(v)] = sorted([(w,c) for w, c in _tb.iteritems()], key=lambda x:x[1], reverse=True)
    return out


def combineCS_replece(big, small):
    out = collections.OrderedDict()
    for v, cs in big.iteritems():
        _tborg = collections.Counter({unicode(t[0]):t[1] for t in cs})
        _tb = collections.Counter({unicode(t[0]):t[1] for t in small[v]}) if v in small else _tborg
        out[unicode(v)] = sorted([(w,c) for w, c in _tb.iteritems()], key=lambda x:x[1], reverse=True)
    return out