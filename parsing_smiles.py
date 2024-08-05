import re

class MoleculeParser:
    """
    A class to parse and interpret chemical species strings, particularly focusing on Carbon atoms
    and their bonds, based on a simplified SMILES notation.

    Attributes:
        species (str): The string representation of the chemical species.
        atoms (list): A list to store parsed atomic elements.
        symbols (list): A list to store parsed bond symbols.
        branches_bond (bool): Indicator for the presence of branching bonds.
        cyclic_bond (bool): Indicator for the presence of cyclic bonds.
        diff_dir_branch (bool): Indicator for the presence of different directional branches.

    Methods:
        parse_species(): Parses the species string to extract atomic elements and bonds.
        special_cases(): Handles special cases not following the standard SMILES notation.
        handle_standard_cases(): Processes standard cases for Carbon atoms.
        handle_subscripts(): Processes atoms with subscripts to handle special bonding scenarios.
        parse_symbols_in_brackets(): Parses symbols within parentheses.
        check_c_in_brackets(): Checks if Carbon atoms are within brackets.
        bracket_handling(): Handles bonds and atoms around parentheses.
        bracket_handling_continued(): Continues handling bonds and atoms around parentheses.
        h_in_brackets(): Handles Hydrogen atoms within brackets.
        process_species(): Processes the species string by invoking various helper methods.
        get_species_details(): Returns the detailed interpretation of the species.
    """
    def __init__(self, species):
        """
        Initializes the MoleculeParser with the given species string.

        Args:
            species (str): The string representation of the chemical species.
        """
        self.species = species
        self.atoms = []
        self.symbols = []

        self.branches_bond = False
        self.cyclic_bond = False
        self.diff_dir_branch = False
    
    def parse_species(self):
        """
        Parses the species string to extract atomic elements and bond symbols.
        """

        # Use a regular expression to find matches of atomic elements and associated symbols
        matches = re.findall(r'([A-Z][a-z]*)([-#=]*)(\d*)\(?([A-Z]?[a-z]*)([-#=]*)\)?', self.species)
        
        for (element, symbol, count, inner_element, inner_symbol) in matches:
            # Convert the count to an integer if it exists, otherwise default to 1
            count = int(count) if count else 1
            # Extend the atoms list with the element repeated 'count' times
            self.atoms.extend([element] * count)
            self.symbols.append(symbol)
            
            # If there are atoms in (), add it to the atoms list too
            if inner_element:
                self.atoms.extend([inner_element])
                self.symbols.append(inner_symbol)
            
            # Check for the presence of cyclic bonds (indicated by '1', if there are more cycles, 
            # more numbers will be introduced in the species string, but the presence of '1' shows the molecule has a cycle)
            if re.search(r'[A-Z]?[a-z]*\d', self.species):
                self.cyclic_bond = True
             # Check for the presence of branches (indicated by '()' in the species string)
            if '(' in self.species and ')' in self.species:
                self.branches_bond = True
            # Check for the presence of branches in a different direction (indicated by multiple '()' in the species string)
            if self.species.count('(') > 1 and self.species.count(')') > 1:
                self.diff_dir_branch = True

    def special_cases(self):
        """
        Handles special cases not following the standard SMILES notation.

        Returns:
            tuple: A tuple containing atoms and symbols arrays for special cases.
        """

        # Special cases not following the standard SMILES notation
        # Which means the Carbon atoms "C" does not have 4 bonds around them

        # For other atoms, in the polarizability data they do not ignore the hydrogen atoms "H" in them
        # because the number of atoms around them is not fixed as Carbon which always has 4 bonds
        if self.species == 'C':
            return ['C', 'H', 'H', 'H', 'H'], ['-', '-', '-', '-']
        elif self.species == 'Zn-N#C':
            return ['Zn', 'C', 'N'], ['-', '#']
        elif self.species == 'C#O':
            return ['C', 'O'], ['#']
        elif self.species == 'Se-C-Se':
            return ['Se', 'C', 'Se'], ['-', '-']
        elif self.species == 'C1-N=N1':
            return ['C', 'N', 'N'], ['-', '=']
        elif self.species == 'C=N=N':
            return ['C', 'N', 'N'], ['=', '=']
        elif self.species == 'Si1-C#C1':
            return ['Si1', 'C', 'C1'], ['-', '#']
        elif self.species == 'C#N-K':
            return ['C', 'N', 'K'], ['#', '-']
        elif self.species == 'F-C-F':
            return ['F', 'C', 'F'], ['-', '-']
        elif self.species == 'C#N-Na':
            return ['C', 'N', 'Na'], ['#', '-']
        elif self.species == 'Al-N#C':
            return ['Al', 'N', 'C'], ['-', '#']
        elif self.species == 'H-C-Cl':
            return ['H', 'C', 'Cl'], ['-', '-', '-']
        elif self.species == 'H-C-Br':
            return ['H', 'C', 'Br'], ['-', '-', '-']
        elif self.species == 'C#C-C=O':
            return ['C', 'C', 'C', 'O'], ['#', '-', '=']
        elif self.species == 'H-O-B-C(-H)-H':
            return ['C', 'H', 'H', 'H', 'B', 'O'], ['-', '-', '-', '-', '-']
        elif self.species == 'C12-O-Li-O1-Li-O2':
            return ['Li', 'Li', 'C', 'O', 'O', 'O'], ['-', '-', '-', '-', '-', '-', '-']
        elif self.species == 'O-C-Fe(-C-O)(-C-O)(-C-O)-C-O':
            return ['Fe', 'C', 'O', 'C', 'O', 'C', 
                    'O', 'C', 'O', 'C', 'O'], ['-', '-', '-', '-', '-', 
                                              '-', '-','-', '-', '-']
        return None, None
    
    def handle_standard_cases(self):
        """
        Processes standard cases for Carbon atoms, ensuring appropriate bond counts.

        Returns:
            tuple: A tuple containing atoms and symbols arrays.
        """
        atoms_array = []
        symbols_array = []

        i = 0
        while i < len(self.atoms):
            atom = self.atoms[i]
            symbol = self.symbols[i] if i < len(self.symbols) else ""
            
            # Since the SMILES notation works for Carbon atom "C" in this code
            if atom == 'C':
                # Identify symbols before and after 'C' 
                prev_symbol = self.symbols[i - 1] if i > 0 else ""

                next_symbol = self.symbols[i + 1] if i + 1 < len(self.symbols) and self.atoms[i + 1] == 'H' else "" 
                if self.species[-1] == ')' or self.species[-1].isdigit():
                    next_symbol = ""

                # Handle symbols in the brackets if they are connected to 'C'
                bracket_symbols = []
                temp_species = self.species

                while '(' in temp_species:
                    start_idx = temp_species.index('(') # Find the start index of '('
                    end_idx = temp_species.index(')')   # Find the end index of ')'
                    bracket_content = temp_species[start_idx + 1:end_idx] # Extract content between '(' and ')'

                    # After extracting, remove processed brackets from 'temp_species'
                    temp_species = temp_species[:start_idx] + temp_species[end_idx + 1:]
                    # Find inner symbols (-, =, #) within the bracket content
                    inner_matches = re.findall(r'([-#=])', bracket_content)
                    if inner_matches and self.atoms[i] == 'C':
                        bracket_symbols.append(inner_matches[0])

                # Since Carbon always has 4 bonds around them, 
                # the number of "-H" attached to them can be calculated as following
                total_bonds = sum(1 if s == '-' else 2 if s == '=' else 3 if s == '#' else 0
                                  for s in [symbol, prev_symbol, next_symbol] + bracket_symbols)
                hydrogen_count = 4 - total_bonds
                
                # Update the atoms array and ensure all H atoms are connected to C
                atoms_array.append(atom)
                for _ in range(hydrogen_count):
                    atoms_array.append('H')
                    symbols_array.append('-')
                symbols_array.append(symbol)
            else:
                atoms_array.append(atom)
                symbols_array.append(symbol)
            i += 1

        return atoms_array, symbols_array
    

    def handle_subscripts(self, atoms_array, symbols_array):
        """
        Processes atoms with subscripts to handle cyclic structures.

        Args:
            atoms_array (list): The list of atomic elements.
            symbols_array (list): The list of bond symbols.

        Returns:
            tuple: A tuple containing updated atoms and symbols arrays.
        """

        # Extract atoms with subscripts and process them separately
        atoms_with_subscript = re.findall(r'([A-Z][a-z]*\d+)', self.species)
        decomp_atoms = []
        
        def handle_atoms_with_subscripts(atoms_with_subscript):
            # Find the atoms to be added into the atom_array and the atoms to be calculated
            for subscript_group in atoms_with_subscript:
                atom_to_add, subscripts = re.match(r'([A-Z][a-z]*)(\d+)', subscript_group).groups()
                for _ in subscripts:
                    decomp_atoms.append(atom_to_add)
                

                # Remove "-H" from the atoms_array and symbols_array based on the number of "C"
                if 'C' in decomp_atoms:
                    for _ in range(decomp_atoms.count('C')):
                        if 'H' in atoms_array:
                            atoms_array.remove('H')
                            symbols_array.remove('-')
            
            # Handle the case that all atoms are Carbon in the molecule
            # and there is only 1 cycle
            all_carbon = all(atom == 'C' for atom in self.atoms)
            if all_carbon == True and subscripts == '1':
                atoms_array.remove('H')
                symbols_array.remove('-')
            
            # For aromatic molecules
            if self.atoms.count('C') >= 6:
                if '=' in self.species:
                  if atoms_array.count('H') >= 2:
                    for _ in range(2):
                        atoms_array.remove('H')
                  if symbols_array.count('-') >= 3:
                    for _ in range(3):
                        symbols_array.remove('-')
                    symbols_array.append('=')

            # Connect atoms with numbers after them with a '-'
            for _ in range(len(decomp_atoms)):
                symbols_array.append('-')
         
        if atoms_with_subscript:
            handle_atoms_with_subscripts(atoms_with_subscript)
        
        # Remove empty elements ('') from symbols_array
        symbols_array = [symbol for symbol in symbols_array if symbol]

        return atoms_array, symbols_array
    
    
    def parse_symbols_in_brackets(self, symbols_array):
        """
        Parses symbols within parentheses and append them to the symbols array.

        Args:
            symbols_array (list): The list of bond symbols.
        """
        spec = self.species
        while '(' in spec:
            # Find the index of the first '('
            start_idx = spec.index('(')

            # Find the corresponding closing parenthesis, which means if there is a "(" after "("
            # without a ")" in the middle, the corresponding ")" to the first "(" is the second ")"
            # this logic remains the same when there is 3 or more "(" in a row
            end_idx = start_idx
            open_count = 1
            while open_count > 0 and end_idx < len(spec) - 1:
                end_idx += 1
                if spec[end_idx] == '(':
                    open_count += 1
                elif spec[end_idx] == ')':
                    open_count -= 1

            # Raise an error if the number of "(" and ")" are not the same
            if open_count != 0:
                raise ValueError("Unbalanced parentheses")
            
            # Extract content between '(' and ')'
            bracket_content = spec[start_idx + 1: end_idx]
            # Remove processed brackets from 'spec'
            spec = spec[:start_idx] + spec[end_idx + 1:]

            inner_matches = re.findall(r'([-#=])', bracket_content)
            symbols_array.extend(inner_matches)
    
    def check_all_c_in_brackets(self):
        """
        Checks if Carbon atoms are within brackets.

        Returns:
            bool: True if Carbon is within brackets, False otherwise.
        """
        # If there is no bracket, Carbon atom "C" is obviously not in brackets
        if '(' not in self.species or ')' not in self.species:
            return False

        stack = []
        positions = []

        # Iterate through each character and track positions of parentheses
        for i, char in enumerate(self.species):
            if char == '(':
                stack.append(i) # Push the index of '(' onto the stack
            elif char == ')':
                if stack:
                    start_idx = stack.pop() # Pop the index of the matching '(' from the stack
                    end_idx = i
                    # Store the start and end indices of the bracket pair "(", ")"
                    positions.append((start_idx, end_idx))

        for start_idx, end_idx in positions:
            # Check if 'C' is present as a standalone word within the substring between the brackets
            if 'C' not in re.findall(r'\bC\b', self.species[start_idx:end_idx]):
                return False
        return True
    
    def bracket_handling(self, atoms_array, symbols_array):
        """
        Handle the contents in the brackets
        """
        
        # Find the first occurence of "("
        first_opening_paren_idx = self.species.find('(')
        # Find the last occurence of ")"
        last_closing_paren_idx = self.species.rfind(")")

        # Find the symbols after the last ')', since it may be connected with the Carbon atom "C"
        # before the bracket, thus affecting the number of "-H" in the molecule
        if '(' in self.species and ')' in self.species:
            if last_closing_paren_idx != -1 and last_closing_paren_idx + 1 < len(self.species):
              if self.species[first_opening_paren_idx - 1] == 'C':
               for char in self.species[last_closing_paren_idx + 1]:
                   if char in '-#=':
                        symbols_array.append(char)
                        bond_to_remove = sum(1 if s == '-' else 2 if s == '=' else 3 if s == '#' else 0
                                      for s in char)
                        for _ in range(bond_to_remove):
                           if 'H' in atoms_array:
                               atoms_array.remove('H')
                               symbols_array.remove('-')

    
        # Find the first symbol appears in the bracket, since it may be connected with the Carbon atom "C"
        # before the bracket, thus affecting the number of "-H" in the molecule
        if '(' in self.species and ')' in self.species:
            if first_opening_paren_idx > 0:
                if self.species[first_opening_paren_idx - 1] != 'C':
                    for _ in range(self.species.count('(') - 1):
                        symbols_array.remove('-')


    def bracket_handling_continued(self, atoms_array, symbols_array):
        """
        Handle the contents in the brackets
        """

        # Recall the variables in previous methods
        first_opening_paren_idx = self.species.find('(')
        c_in_brackets = self.check_all_c_in_brackets()
        
        # When C is not in the brackets
        if c_in_brackets == False:
            if '(' in self.species and ')' in self.species:
              # If the atom before the first "(" is Carbon, the symbol in the bracket should only affect
              # itself, but not all Carbons in the species
              if self.species[first_opening_paren_idx - 1] == 'C':
                if self.species[first_opening_paren_idx + 1] == '-':
                    for _ in range(self.atoms.count('C') - 1):
                        atoms_array.append('H')
                        symbols_array.append('-')

                if self.species[first_opening_paren_idx + 1] == '=':
                    for _ in range(2 * self.atoms.count('C') - 2):
                        atoms_array.append('H')
                        symbols_array.append('-')
            
                if self.species[first_opening_paren_idx + 1] == '#':
                    for _ in range(3 * self.atoms.count('C') - 3):
                        atoms_array.append('H')
                        symbols_array.append('-')

        else:
            # For the case 'C' is in the brackets
            spec = self.species
            original_spec = self.species
            while '(' in spec:
                # Find the index of the first '('
                start_idx = spec.index('(')
    
               # If there's a "C" before the first '('
                if start_idx > 0 and spec[start_idx - 1] == 'C':
                    # Find the corresponding closing parenthesis
                    end_idx = start_idx
                    open_count = 1
                    while open_count > 0 and end_idx < len(spec) - 1:
                        end_idx += 1
                        if spec[end_idx] == '(':
                           open_count += 1
                        elif spec[end_idx] == ')':
                           open_count -= 1
                        
                        # append the number of "H" and "-" according to the C and symbol in the bracket
                        if spec[start_idx + 1] == '-':
                           for _ in range(3):
                               atoms_array.append('H')
                               symbols_array.append('-')

                        if spec[start_idx + 1] == '=':
                           for _ in range(2):
                               atoms_array.append('H')
                               symbols_array.append('-')

                        if spec[start_idx + 1] == '#':
                           for _ in range(1):
                               atoms_array.append('H')
                               symbols_array.append('-')
                       
                        # Remove the parentheses and the enclosed substring
                        spec = spec[:start_idx - 1] + spec[end_idx + 1:]
                   
                else:
                    break
                if original_spec.count('(') == 1:
                  if atoms_array.count('H') >= 9:
                    for _ in range(8):
                        atoms_array.remove('H')
                        symbols_array.remove('-')
                if original_spec.count('(') == 2:
                    if original_spec.count('C') == 5:
                        for _ in range(2):
                            atoms_array.remove('H')
                            symbols_array.remove('-')
                    if original_spec.count('C') >= 6:
                        for _ in range(11 - original_spec.count('C')):
                            atoms_array.remove('H')
                            symbols_array.remove('-')
            else:
                # When the atom just before the parenthesis "(" is not Carbon, 
                # the symbol in the bracket and the symbol after ")" do not affect the hydrogen count
                if self.species[first_opening_paren_idx + 1] == '-':
                    for _ in range(self.atoms.count('C') - 1):
                        atoms_array.append('H')
                        symbols_array.append('-')
                    
                if self.species[first_opening_paren_idx + 1] == '=':
                    for _ in range(2 * self.atoms.count('C') - 2):
                        atoms_array.append('H')
                        symbols_array.append('-')
                    
                if self.species[first_opening_paren_idx + 1] == '#':
                    for _ in range(3 * self.atoms.count('C') - 3):
                        atoms_array.append('H')
                        symbols_array.append('-')
                
    
    def h_in_brackets(self, atoms_array, symbols_array):
        """
        Check whether Hydrogen "H" is in the brackets
        """
        
        # Recall the variables
        first_opening_paren_idx = self.species.find('(')
        last_closing_paren_idx = self.species.rfind(")")

        if '(' in self.species and ')' in self.species:
            spec = self.species[first_opening_paren_idx + 1:last_closing_paren_idx]
            # Make sure Hydrogen is in the species
            if 'H' in spec:

              # If there are Carbon in front of the bracket, add "-H" back based on the number
              # of Carbon + 1 in the species, since by default they will be removed
              if 'C' in self.species:
                if 'C' in self.species[:first_opening_paren_idx]:
                    for _ in range(self.species[:first_opening_paren_idx].count('C') + 1):
                        atoms_array.append('H')
                        symbols_array.append('-')

                if 'C' in self.species[last_closing_paren_idx:]:
                    if self.species[last_closing_paren_idx:].count('C') == 1:
                        atoms_array.remove('H')
                        symbols_array.remove('-')

                    if self.species[last_closing_paren_idx:].count('C') >= 3:
                        for _ in range(self.species[last_closing_paren_idx].count('C') - 2):
                            atoms_array.append('H')
                            symbols_array.append('-')
                        
                # If there is no Carbon before or after the bracket
                if 'C'  not in self.species[:first_opening_paren_idx] and 'C' not in self.species[last_closing_paren_idx:]:
                    for _ in range(spec.count('(') + spec.count('C') - 2):
                        atoms_array.append('H')
                        symbols_array.append('-')
    
                    
    def process_species(self):
        """
        Process the species string to extract atoms and symbols, handling special cases.

        Returns:
            array: the atoms_array and symbols_array
        """
        special_atoms, special_symbols = self.special_cases()
        if special_atoms and special_symbols:
            return special_atoms, special_symbols

        self.parse_species()
        atoms_array, symbols_array = self.handle_standard_cases()
        atoms_array, symbols_array = self.handle_subscripts(atoms_array, symbols_array)
        self.parse_symbols_in_brackets(symbols_array)
        self.bracket_handling(atoms_array, symbols_array)
        self.bracket_handling_continued(atoms_array, symbols_array)
        self.h_in_brackets(atoms_array, symbols_array)

        return atoms_array, symbols_array
    
    def get_species_details(self):
        """
        Get details of the parsed species including atoms, symbols, and special conditions.

        Returns:
            tuple: the atoms_array and symbols_array, as well as the boolean values for
            branches_bond_bool, cyclic_bond_bool, diff_dir_branch_bool
        """
        atoms_array, symbols_array = self.process_species()
        branches_bond_bool = True if self.branches_bond else False
        cyclic_bond_bool = True if self.cyclic_bond else False
        diff_dir_branch_bool = True if self.diff_dir_branch else False

        return atoms_array, symbols_array, branches_bond_bool, cyclic_bond_bool, diff_dir_branch_bool

