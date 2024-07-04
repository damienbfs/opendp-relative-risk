from tradeoff import tradeoffCurve

def posterior(alpha, tradeoff, prior):
    if isinstance(tradeoff, tradeoffCurve):
        beta = tradeoff(alpha)
    else:
        beta = tradeoff(alpha)

    posterior = (prior * (1 - beta)) / ((1 - prior)*alpha + prior * (1 - beta))

    return posterior