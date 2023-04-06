import random

jogadores = input("Quem vem ao jogo? ")
while len(set(jogadores.split(" "))) != 10:
    print("Não podem haver jogadores repetidos, tem de ser 10 jogadores diferentes")
    jogadores = input("Quem vem ao jogo? ")
else:
    pass    
while len(jogadores.split(" "))<10 or len(jogadores.split(" "))>10:
    print("Tem de ser 10 jogadores")
    jogadores = input("Quem vem ao jogo? ")
else:
    print("Vou fazer as equipas")
EquipaA = []
EquipaB = []
for jogador in jogadores:
    while len(EquipaA) < 5:
        Next=random.choice(jogadores.split(" "))
        if Next in EquipaA:
            Next=random.choice(jogadores.split(" "))
        else:
            EquipaA.append(Next)        
    else:
        while len(EquipaB) <5:
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

    

