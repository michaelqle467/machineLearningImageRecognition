def card_value(c):
    rank = extract_rank(c)
    if rank in ['J','Q','K']:
        return 10
    if rank == 'A':
        return 11
    return int(rank)

def extract_rank(class_name: str) -> str:
    """
    Converts '10 Hearts' -> '10'
    Converts 'Q Spades' -> 'Q'
    Converts 'A Diamonds' -> 'A'
    """
    return class_name.split()[0]


def hand_value(cards):
    total, aces = 0, 0
    for c in cards:
        rank = extract_rank(c)
        if rank == 'A':
            aces += 1
            total += 11
        elif rank in ['J','Q','K']:
            total += 10
        else:
            total += int(rank)

    while total > 21 and aces:
        total -= 10
        aces -= 1

    return total

def strategy(player_cards, dealer_upcard, allow_double=True, allow_surrender=True):
    """Returns one of: HIT, STAND, DOUBLE, SPLIT, SURRENDER"""
    total = hand_value(player_cards)
    d = card_value(dealer_upcard)

    # Surrender (common basic rule-of-thumb; tweak per casino rules)
    if allow_surrender and total == 16 and d in [9, 10, 11]:
        return "SURRENDER"
    if allow_surrender and total == 15 and d == 10:
        return "SURRENDER"

    # Pair splitting (minimal; extend as needed)
    if len(player_cards) == 2 and player_cards[0] == player_cards[1]:
        if player_cards[0] in ['A', '8']:
            return "SPLIT"

    # Soft hands
    if 'A' in player_cards and total <= 21:
        if total >= 19:
            return "STAND"
        if total == 18 and d in [2, 7, 8]:
            return "STAND"
        # simple double guidance on soft totals
        if allow_double and total in [13,14] and d in [5,6]:
            return "DOUBLE"
        if allow_double and total in [15,16] and d in [4,5,6]:
            return "DOUBLE"
        if allow_double and total in [17,18] and d in [3,4,5,6]:
            return "DOUBLE"
        return "HIT"

    # Hard hands
    if total >= 17:
        return "STAND"
    if total <= 8:
        return "HIT"
    if allow_double and total == 9 and d in [3,4,5,6]:
        return "DOUBLE"
    if allow_double and total == 10 and d in [2,3,4,5,6,7,8,9]:
        return "DOUBLE"
    if allow_double and total == 11 and d in [2,3,4,5,6,7,8,9,10,11]:
        return "DOUBLE"
    if total == 12 and d in [4,5,6]:
        return "STAND"
    if 13 <= total <= 16 and d <= 6:
        return "STAND"

    return "HIT"
