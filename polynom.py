import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import newton

class Polynom:
    def __init__(self, degree, coefficients):
        """
        Initierar ett polynom med ett givet gradtal och koefficienter.
        degree: Graden av polynomet (t.ex. 3 för x^3)
        coefficients: Lista med koefficienter där indexet motsvarar graden av termen.
        """
        self.degree = degree
        self.coefficients = coefficients[::-1]  # Reversera listan så att index 0 motsvarar x^0
    
    def __str__(self):
        """
        Skriv ut polynomet på ett snyggt sätt, i fallande ordning av exponenter.
        """
        terms = []
        for i, c in enumerate(self.coefficients):
            if c == 0:
                continue
            elif i == 0:
                terms.append(f"{c}")
            elif i == 1:
                terms.append(f"{c}x")
            else:
                terms.append(f"{c}x^{i}")
        
        # Omvänd ordning så högsta exponenten kommer först
        return " + ".join(reversed(terms)) if terms else "0"


    def __call__(self, x):
        """
        Evaluera polynomet för ett givet x.
        """
        return sum(c * x**i for i, c in enumerate(self.coefficients))

    def __add__(self, other):
        """
        Addition av polynom: Adderar koefficienterna för två polynom.
        """
        # Justera längden på koefficientlistorna genom att lägga till nollor där det behövs
        max_len = max(len(self.coefficients), len(other.coefficients))
        
        # Kopiera koefficientlistorna och lägga till nollor om det behövs
        self_coeffs = self.coefficients + [0] * (max_len - len(self.coefficients))
        other_coeffs = other.coefficients + [0] * (max_len - len(other.coefficients))
        
        # Addera koefficienterna för varje grad
        result_coefficients = [a + b for a, b in zip(reversed(self_coeffs), reversed(other_coeffs))]
        
        # Skapa ett nytt polynom med de adderade koefficienterna
        return Polynom(len(result_coefficients) - 1, result_coefficients)

    def __sub__(self, other):
        """
        Subtraktion av polynom: Subtraherar koefficienterna för två polynom.
        """
        # Justera längden på koefficientlistorna genom att lägga till nollor där det behövs
        max_len = max(len(self.coefficients), len(other.coefficients))
        
        # Kopiera koefficientlistorna och lägga till nollor om det behövs
        self_coeffs = self.coefficients + [0] * (max_len - len(self.coefficients))
        other_coeffs = other.coefficients + [0] * (max_len - len(other.coefficients))
        
        # Subtrahera koefficienterna för varje grad
        result_coefficients = [a - b for a, b in zip(reversed(self_coeffs), reversed(other_coeffs))]
        
        # Skapa ett nytt polynom med de subtraherade koefficienterna
        return Polynom(len(result_coefficients) - 1, result_coefficients)

    def __mul__(self, other):
        """
        Multiplicering av polynom: Multiplicera koefficienterna för två polynom.
        """
        # Resultatets grad är summan av gradtalen från de två polynomen
        result_degree = self.degree + other.degree
        
        # Skapa en lista med nollor för att hålla alla koefficienter
        result_coefficients = [0] * (result_degree + 1)

        # Multiplicera varje term från det första polynomet med varje term från det andra polynomet
        for i, a in enumerate(self.coefficients):
            for j, b in enumerate(other.coefficients):
                # Rätt index beräknas genom att summera exponenterna
                index = i + j
                # Lägg till produkten av termerna på rätt position i resultatkoefficienterna
                result_coefficients[index] += a * b

        # Skapa och returnera ett nytt Polynom med de resulterande koefficienterna
        return Polynom(result_degree, result_coefficients)

    def __eq__(self, other):
        """
        Jämför om två polynom är lika.
        """
        return self.coefficients == other.coefficients

    def diff(self):
        """
        Derivera polynomet.
        """
        # Derivera varje term c_i * x^i genom att multiplicera koefficienten med i och
        # reducera exponenten (ignorera den konstanta termen x^0).
        result_coefficients = [i * c for i, c in enumerate(self.coefficients)][1:]

        # Skapa ett nytt polynom med de deriverade koefficienterna.
        return Polynom(self.degree - 1, result_coefficients)

    def plot(self, x_min=-10, x_max=10, num_points=400):
        """
        Plotta polynomet inom ett givet x-intervall.
        """
        x_values = np.linspace(x_min, x_max, num_points)
        y_values = [self(x) for x in x_values]
        plt.plot(x_values, y_values)
        plt.xlabel('x')
        plt.ylabel('p(x)')
        plt.title(f"Plot of Polynomial: {self}")
        plt.grid(True)
        plt.show()

    def zero(self, initial_guess=0):
        """
        Hitta ett nollställe (rot) till polynomet med hjälp av Newton-Raphson.
        """
        # Derivata av polynomet för att använda i Newton-Raphson-metoden
        def derivative(x):
            return self.diff()(x)

        # Newton-Raphson för att hitta nollstället
        root = newton(self.__call__, initial_guess, fprime=derivative)
        return root

    def integrate(self, a, b):
        """
        Integrera polynomet mellan a och b genom att skapa den primitiva funktionen.
        """
        # Skapa koefficienterna för den primitiva funktionen genom att dividera varje koefficient med (i+1).
        integral_coefficients = [c / (i + 1) for i, c in enumerate(self.coefficients)]

        # Skapa ett polynom för den primitiva funktionen med de nya koefficienterna.
        integral_polynom = Polynom(self.degree + 1, integral_coefficients)

        # Evaluera den primitiva funktionen vid gränserna a och b.
        result_b = integral_polynom(b)  # Evaluera vid b
        result_a = integral_polynom(a)  # Evaluera vid a

        # Returnera skillnaden mellan resultatet vid b och a för att få den bestämda integralen.
        return result_b - result_a

# Skapa två polynom
p1 = Polynom(3, [3, 0, 1, 5])  # p1 = 3x^3 + x + 5
print(f"p1: {p1}")

p2 = Polynom(2, [1, 1, 0])     # p2 = x^2 + x
print(f"p2: {p2}")

# Addition av polynom
p3 = p1 + p2  # p3 = 3x^3 + x^2 + 2x + 5 
print(f"p3: {p3}")

# Subtraktion av polynom
p4 = p1 - p2  # p4 = 3x^3 - x^2 + 5 
print(f"p4: {p4}")

# Multiplicering av polynom
p5=p1*p2 # p5 = 3x^5 + 3x^4 + x^3 + 6x^2 + 5x
print(f"p5: {p5}")

# Derivering av polynom
p6 = p1.diff()  # p6 = 9x^2 + 1  #FEL!!!!
print(f"p6 (derivata av p1): {p6}")

# Jämför om polynomen är lika
print(f"p1 == p2: {p1 == p2}")

# Utvärdera p1 för x = 2
val = p1(2)  # p1(2) = 3*2^3 + 2 + 5 = 31
print(f"p1(2): {val}")

# Plotta polynomet p1
p1.plot(x_min=-10, x_max=10)

# Hitta ett nollställe till p1
x0 = p1.zero(0)   #Nollställe: -1.0921268117651426
print(f"Nollställe för p1: {x0}")

# Integrera polynomet p1 mellan 0 och 1
I = p1.integrate(0, 1)  # Integralen 0 till 1: 6.2500000 #FEL!!!!
print(f"Integralen av p1 från 0 till 1: {I}")