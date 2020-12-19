# -*- coding: utf-8 -*-
# Very simple unit support:
#  - creates variable names like 'mV' and 'kHz'
#  - the value assigned to the variable corresponds to the scale prefix
#    (mV = 0.001)
#  - the actual units are purely cosmetic for making code clearer:
#
#    x = 20*pA    is identical to    x = 20*1e-12
#
# No unicode variable names (μ,Ω) allowed until python 3, but just assigning
# them to the globals dict doesn't error in python 2.
import unicodedata

# All unicode identifiers get normalized automatically
SI_PREFIXES = unicodedata.normalize("NFKC", "yzafpnµm kMGTPEZY")
UNITS = unicodedata.normalize("NFKC", "m,s,g,W,J,V,A,F,T,Hz,Ohm,Ω,S,N,C,px,b,B,Pa").split(",")
allUnits = {}


def addUnit(prefix, val):
    g = globals()
    for u in UNITS:
        g[prefix + u] = val
        allUnits[prefix + u] = val


for pre in SI_PREFIXES:
    v = SI_PREFIXES.index(pre) - 8
    if pre == " ":
        pre = ""
    addUnit(pre, 1000 ** v)

addUnit("c", 0.01)
addUnit("d", 0.1)
addUnit("da", 10)
addUnit("h", 100)
# py2 compatibility
addUnit("u", 1e-6)


def evalUnits(unitStr):
    """
    Evaluate a unit string into ([numerators,...], [denominators,...])
    Examples:
        N m/s^2   =>  ([N, m], [s, s])
        A*s / V   =>  ([A, s], [V,])
    """
    pass


def formatUnits(units):
    """
    Format a unit specification ([numerators,...], [denominators,...])
    into a string (this is the inverse of evalUnits)
    """
    pass


def simplify(units):
    """
    Cancel units that appear in both numerator and denominator, then attempt to replace 
    groups of units with single units where possible (ie, J/s => W)
    """
    pass
