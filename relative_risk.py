from tradeoff import tradeoffCurve

def posterior(alpha, tradeoff, prior):
    beta = tradeoff(alpha)
    posterior = (prior * (1 - beta)) / ((1 - prior)*alpha + prior * (1 - beta))

    return posterior


def relative_risk(alpha, tradeoff, prior):
    beta = tradeoff(alpha)
    relative_risk = (1 - beta) / ((1 - prior)*alpha + prior * (1 - beta))

    return relative_risk