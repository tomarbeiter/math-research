"""Class to handle generating the matrices for T^n"""
import copy
import numpy as np
import itertools as it
from pymongo import MongoClient
import json


class Torus:
    def __init__(self, n):
        self.n = n
        self.total = 0
        self.base_matrices = {
            "stick_rotations": [],
            "stick_reflections": [],
            "re_all": [],
            "stickless_reflections": []
        }
        self.base_labels = []
        self.labels_dict = {
            "1": "A",
            "2": "B",
            "3": "C",
            "4": "D",
            "5": "E",
            "6": "F",
            "7": "G",
            "8": "H",
            "9": "I",
            "10": "J"
        }
        self.composed_matrices = []
        
        # Generate base matrices
        self._stick_rotations()
        self._stick_reflections()
        self._re_all()
        self._stickless_reflections()
        self._compose()
        # Build Cayley Table once we know how big it is
        self.cayley_table = [[0 for _ in range(len(self.composed_matrices))] for _ in range(len(self.composed_matrices))]
        self.total_c = len(self.composed_matrices)
        self._cayley()

        

    def _stick_rotations(self):
        """Generate the matrices for each stick rotation.
        
        Some hackiness to adapt the formula we have to how Python lists work. 
        """
        for k in range (1, self.n + 1):
            rotation = np.array(copy.deepcopy([[0 for _ in range(self.n * 2)] for _ in range (self.n * 2)]))
            # Rows
            i = 1
            for x in range(1, self.n * 2, 2):
                # Columns
                pos = (self.n + i - k) % self.n if (self.n + i - k) % self.n != 0 else self.n
                i += 1
                for y in range(1, self.n * 2, 2):
                    # Convert position from formula to 2nx2n matrix
                    if (pos * 2) - 1 == y:
                        rotation[x - 1][y - 1] = 1
                        rotation[x][y] = 1
            # Create label for rotation
            if k == self.n:
                self.base_labels.insert(0, "Identity")
            else:
                self.base_labels.append("r0-z"+str(360//self.n * k))
            # Append rotation
            self.base_matrices["stick_rotations"].append(rotation)
        # Count them up and move the identity to the front
        self.total += len(self.base_matrices["stick_rotations"])
        self.base_matrices["stick_rotations"].insert(0, self.base_matrices["stick_rotations"].pop())

    def _stick_reflections(self):
        """Generate the matrices for each stick reflection (r_0-A_i) (routing code)."""
        if self.n % 2:
            self._odd_stick_reflections()
        else:
            self._even_stick_reflections()

    def _odd_stick_reflections(self):
        """Generate matrices for r_0-A_i when n is odd."""
        for i in range (1, self.n + 1):
            # reset matrix to 0's for each iteration
            reflection = np.array(copy.deepcopy([[0 for _ in range(self.n * 2)] for _ in range (self.n * 2)]))
            # j is a double row increment, k is single row to track with i
            for j, k in zip(range(1, self.n * 2 + 1, 2), range(1, self.n + 1)):
                # position in n x n matrix
                pos = (2 * i - k) % self.n
                # convert position to 2n x 2n
                pos = pos * 2 - 1
                reflection[j][pos] = -1
                reflection[j - 1][pos - 1] = -1
            # Create labels for each reflection
            self.base_labels.append("r0-"+self.labels_dict[str(i)])
            self.base_matrices["stick_reflections"].append(reflection)
        self.total += len(self.base_matrices["stick_reflections"])

    def _even_stick_reflections(self):
        """Generate matrices for r_0-A_i and r_0-A_i/A_i+1 when n is even."""
        div = False
        i = 1
        x = 1
        while i <= self.n // 2:
            # reset matrix to 0's for each iteration
            reflection = np.array(copy.deepcopy([[0 for _ in range(self.n * 2)] for _ in range (self.n * 2)]))
            if not div:
                # j is a double row increment, k is single row to track with i
                for j, k in zip(range(1, self.n * 2 + 1, 2), range(1, self.n + 1)):
                    # position in n x n matrix
                    pos = (2 * i - k) % self.n
                    # convert position to 2n x 2n
                    pos = pos * 2 - 1
                    reflection[j][pos] = -1
                    reflection[j - 1][pos - 1] = -1
                div = True
                # Create labels for each reflection
                self.base_labels.append("r0-"+self.labels_dict[str(x)])
                self.base_matrices["stick_reflections"].append(reflection)
            else:
                for j, k in zip(range(1, self.n * 2 + 1, 2), range(1, self.n + 1)):
                    # position in n x n matrix
                    pos = (2 * i + 1 - k) % self.n
                    # convert position to 2n x 2n
                    pos = pos * 2 - 1                   
                    reflection[j][pos] = -1
                    reflection[j - 1][pos - 1] = -1
                div = False
                i += 1
                self.base_labels.append("r0-"+self.labels_dict[str(x)])
                self.base_matrices["stick_reflections"].append(reflection)
            x += 1
        self.total += len(self.base_matrices["stick_reflections"])

    def _re_all(self):
        """Generate matrix for re-all alternating 1 and -1 on diagonal."""
        re_all = np.array(copy.deepcopy([[0 for _ in range(self.n * 2)] for _ in range (self.n * 2)]))
        for i in range(0, self.n * 2):
            if not i % 2:
                re_all[i][i] = 1
            else:
                re_all[i][i] = -1 
        # Create a label
        self.base_labels.append("re-all")
        self.base_matrices["re_all"].append(re_all)
        self.total += 1

    def _stickless_reflections(self):
        """Generate r1-A_i reflections -1's at the corresponding i block, 1's on the rest of the diagonal."""
        for i in range(1, self.n + 1):
            reflection = np.array(copy.deepcopy([[0 for _ in range(self.n * 2)] for _ in range (self.n * 2)]))
            for j, k in zip(range(1, self.n * 2 + 1, 2), range(1, self.n + 1)):
                if k == i:
                    reflection[j][j] = -1
                    reflection[j - 1][j - 1] = -1
                else:
                    reflection[j][j] = 1 
                    reflection[j - 1][j - 1] = 1 
            # Create labels
            self.base_labels.append("r1-"+self.labels_dict[str(i)])
            self.base_matrices["stickless_reflections"].append(reflection)
        self.total += len(self.base_matrices["stickless_reflections"])

    def _compose(self):
        for matrix in self.base_matrices["stick_rotations"]:
            self.composed_matrices.append(matrix)
        for matrix in self.base_matrices["stick_reflections"]:
            self.composed_matrices.append(matrix)
        # First half
        for i in range(1, self.n + 1):
            # Get all r1 combos
            combos = list(it.combinations(self.base_matrices["stickless_reflections"], i))
            # Multiply each combo of r1 against all 6 bases
            for combo in combos:
                for j in range(self.n * 2):
                    result = self.composed_matrices[j]
                    # Iterate in reverse over the combo
                    for k in range(len(combo) - 1, -1, -1):
                        result = np.matmul(combo[k], result)
                    self.composed_matrices.append(result)
        # Second half
        for i in range(self.n * 2):
            self.composed_matrices.append(np.matmul(self.composed_matrices[i], self.base_matrices["re_all"][0]))
        for i in range(1, self.n + 1):
            # Get all r1 combos
            combos = list(it.combinations(self.base_matrices["stickless_reflections"], i))
            # Multiply each combo of r1 against all 6 bases
            for combo in combos:
                for j in range(self.n * 2):
                    # only difference is we stack re-z on all the combos from the first half
                    result = self.composed_matrices[j]
                    # Iterate in reverse over the combo
                    for k in range(len(combo) - 1, -1, -1):
                        result = np.matmul(combo[k], result)
                    result = np.matmul(self.base_matrices["re_all"][0], result)
                    self.composed_matrices.append(result)

    def _cayley(self):
        """Generate Cayley Table using indices of matrices."""
        for x in range(len(self.composed_matrices)):
            for y in range(len(self.composed_matrices)):
                result = np.matmul(self.composed_matrices[x], self.composed_matrices[y])
                for i in range(len(self.composed_matrices)):
                    if np.array_equal(result, self.composed_matrices[i]):
                        self.cayley_table[x][y] = i + 1 
                        break
                

    def print_matrix(self, key):
        for i in range(len(self.base_matrices[key])):
            for j in range(self.n * 2):
                print(self.base_matrices[key][i][j])
            print("\n")
    
    def print_cayley(self):
        for i in range(len(self.composed_matrices)):
            print(self.cayley_table[i], "\n")

    def print_to_file(self, file):
        for i in range(len(self.composed_matrices)):
            self.composed_matrices[i] = self.composed_matrices[i].tolist()

        response = {
            "n": self.n,
            "total_base": self.total,
            "total_composed": self.total_c,
            "composed": self.composed_matrices,
            "cayley": self.cayley_table
        }
        with open(file, 'w') as outfile:
            json.dump(response, outfile)

torus = Torus(6)
torus.print_cayley()



# torus2 = Torus(3)
# torus2.print_to_file("data/t3.json")

# torus3 = Torus(4)
# torus3.print_to_file("data/t4.json")
