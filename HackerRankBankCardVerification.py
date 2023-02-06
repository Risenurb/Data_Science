cards = int(input())

def is_valid_credit_card(card_number):
    card_number = card_number.replace('-', '')
    for i in range(len(card_number)):
        if card_number[i] == '-':
            if (i == 4 or i == 9 or i == 14):
                card_number = card_number.replace('-', '')
            else:
                return "Invalid"
        elif  card_number[i].isdigit()==False and len(card_number) != 16:
            return "Invalid"
    if len(card_number) != 16:
        return "Invalid"
    if not card_number[0] in ['4', '5', '6']:
        return "Invalid"
    for i in range(len(card_number)-3):
        if card_number[i] == card_number[i+1] == card_number[i+2] == card_number[i+3]:
            return "Invalid"
    return "Valid"

for _ in range(cards):
    card_number = input()
    print(is_valid_credit_card(card_number))