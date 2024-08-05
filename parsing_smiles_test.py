from parsing_smiles import MoleculeParser

species1 = 'C-C#C'
parser1 = MoleculeParser(species1)
details1 = parser1.get_species_details()
print(details1)
print()

species2 = 'C=C(-Cl)-Cl'
parser2 = MoleculeParser(species2)
details2 = parser2.get_species_details()
print(details2)
print()

species3 = 'C1-O-C1'
parser3 = MoleculeParser(species3)
details3 = parser3.get_species_details()
print(details3)
print()

species4 = 'C1-C-C12-C-C2'
parser4 = MoleculeParser(species4)
details4 = parser4.get_species_details()
print(details4)
print()

species5 = 'C-C(-C)-C'
parser5 = MoleculeParser(species5)
details5 = parser5.get_species_details()
print(details5)
print()

species6 = 'F-C(-Cl)(-Cl)-F'
parser6 = MoleculeParser(species6)
details6 = parser6.get_species_details()
print(details6)
print()

species7 = 'C-O-O1'
parser7 = MoleculeParser(species7)
details7 = parser7.get_species_details()
print(details7)
print()