import random

numero = input("Quantas pessoas vem ao jogo? ")
while (int(numero) % 2) != 0:
    print("número de jogadores tem de ser par para haver equipas iguais")
    numero = input("Quantas pessoas vem ao jogo? ")
else:
    pass
jogadores = input("Quem vem ao jogo?\n")

while len(set(jogadores.split(" "))) != int(numero):
    print("Não podem haver jogadores repetidos, tem de ser jogadores diferentes ou não foi introduzido todos os jogadores")
    jogadores = input("Quem vem ao jogo?\n")
else:
    pass   

EquipaA = []
EquipaB = []
for jogador in jogadores:
    while len(EquipaA) < ((int(numero))/2):
        Next=random.choice(jogadores.split(" "))
        if Next in EquipaA:
            Next=random.choice(jogadores.split(" "))
        else:
            EquipaA.append(Next)        
    else:
        while len(EquipaB) <((int(numero))/2):
            Next2=random.choice(jogadores.split(" "))
            if Next2 in EquipaA:
                Next2=random.choice(jogadores.split(" "))
            elif Next2 in EquipaB:
                Next2=random.choice(jogadores.split(" "))
            else:
                EquipaB.append(Next2)
        else:
            break
print(f"{EquipaA} vs {EquipaB}")

    

