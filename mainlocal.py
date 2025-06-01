#type: ignore

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.optimize import curve_fit
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import CubicSpline
from typing import Any

from numba import jit, njit
import numba as nb

# IMPORTANTE

DEBUG_MODE = False

class SimError(Exception):
    pass

# Initial variables

J = 1
H = 0
k_b = 1

@njit
def fastDeltaE(x: int, y:int, spins:np.array, neighbors:np.array) -> int:

    """
    ## ΔE ao inverter o spin da célula (x, y) numa grid de largura L e altura igual

    Nota: A fórmula para calcular a variação de energia ao inverter o spin em (x, y) é dada por

    ΔE = -2 * s * sum(neighborSpins), em que s é o spin central.

    Depois basta multiplicar por -J, pois a energia é por definição -J * ∑_pares_vizinhos - h * ∑_todos

    => ΔE = 2 * J * s * sum(neighborSpins)
    """

    s             = spins[x,y]
    neighborSpins = [spins[x_n,y_n] for x_n,y_n in neighbors[x,y]]

    if H == 0:
        return 2 * J * s * sum(neighborSpins) # Só para não termos de calcular np.sum(self.spins)
    return     2 * s * (J * sum(neighborSpins) + H)

@njit
def fastDeltaMag(x: int, y:int, spins:np.array) -> int:

    """
    ## ΔM ao inverter o spin da célula (x, y) numa grid de largura L e altura igual

    Nota: A magnetização é mais simples, sendo dada por -J * ∑_todos

    A variação de ∑_todos é de -2s, em que s é o spin central (antes era 1, passa a ser -1, variação -2, e vice-versa). Multiplicando por -J,

    ΔM = 2 * J * self.spins[x,y]
    """

    return 2 * J * spins[x,y]

# Classe do Model de Ising
class IsingModel:

    """

    TIPOS (porque não podemos alterar)

    ##### Fields

    L:int
    T:int

    spins:          np array
    energies:       np array
    magnetizations: np array
    systemEnergy:   int

    ##### Métodos

    calc_ener_spin  (self, i:int, j:int) -> float
    calc_ener       (self)               -> float
    calc_mag        (self)               -> float
    iter_monte_carlo(self, n_iter:int)   -> None
    energy          (self)               -> np array
    magnetization   (self)               -> np array


    """

    def __init__( self, L, T ):

        # parametros do modelo
        self.L = L
        self.T = T

        # array com os spins
        self.spins = np.ones( (L, L) )

        # arrays para guardar evolução das variáveis
        self.energies = []
        self.magnetizations = []

        # Inicializa a energia e magnetização do sistema uma única vez, as seguintes são obtidas somando o delta

        self.systemEnergy        = - L ** 2 * (2 * J + H)
        self.systemMagnetization = - J * L ** 2

        self.rng = np.random.default_rng()      # Cria um gerador de números aleatórios para o modelo

        self.ultraConstant = -1/(k_b*self.T)    # Pré-cálculos

        self.neighbors = np.empty((L, L, 4, 2), dtype=int)

        # Pré-calculo dos neighbors
        for i, j in np.ndindex(L, L):
            self.neighbors[i, j] = self.__getNeighborCoordinates(i, j)

        if DEBUG_MODE: self.deltaE_difs = [self.systemEnergy - self.__calcSystemEnergy()]

    def __getNeighborCoordinates(self, x:int, y:int) -> tuple[tuple[int,int,int,int], tuple[int,int,int,int]]:

        """
        Coordenadas dos neighbors de (x, y) numa grid de largura L e altura igual, tendo em atenção a periodicidade.
        ex: getNeighborCoordinates(0, 0, 2) = ((1, 0), (1, 0), (0, 1), (0, 1))
        """

        L = self.L
        return np.array([
            [(x + 1) % L, y],
            [(x - 1) % L, y],
            [x, (y + 1) % L],
            [x, (y - 1) % L],
        ])

        # L = self.L
        # return np.array([
        #     [(x + 1) % L, (x - 1) % L, x, x],
        #     [y,y, (y + 1) % L, (y - 1) % L]
        # ])

        # Equivalente a
        #return (((x + 1) % self.L, y),
        #        ((x - 1) % self.L, y),
        #        (x, (y + 1) % self.L),
        #        (x, (y - 1) % self.L))

    def __calcSystemEnergy(self) -> float:

        """
        ## Calcula a energia atual do sistema

        (este método serve só para testar se E + dE = E* <=> A variação retornada por changeSpin está calculada corretamente)

        Nota: Dividir por 2 é para não contar os pares duas vezes ((i, j) e (j, i)). Isto é seguro pois, devido às condições periódicas, é garantido que existem sempre dois pares iguais, mesmo nas "bordas". (Não é necessário registar os pares já "visitados")
        """

        visited:set[set[tuple[int, int], tuple[int, int]]] = []
        sum = 0
        for index, value in np.ndenumerate(self.spins):
            for n in self.__getNeighborCoordinates(*index):
                if set((tuple(index), tuple(n))) in visited: continue
                sum += value * self.spins[*n]
                visited.append(set((tuple(index), tuple(n))))
                # s = sum([-J*sum(self.spins[*self.neighbors[*index]] * value) * 0.5 - H * value for index, value in np.ndenumerate(self.spins)])

        sum *= -J
        return sum

    def __calcSystemMagnetization(self) -> float:

        """
        ## Calcula a magnetização atual do sistema

        (este método serve só para testar se M + dM = M* <=> A variação está calculada corretamente)
        """

        return - J * np.sum(self.spins)

    # Não usado em iter_monte_carlo, mas pode ser útil para outras simulações
    def deltaE(self, x: int, y:int) -> int:

        """
        ## ΔE ao inverter o spin da célula (x, y) numa grid de largura L e altura igual

        Nota: A fórmula para calcular a variação de energia ao inverter o spin em (x, y) é dada por

        ΔE = -2 * s * sum(neighborSpins), em que s é o spin central.

        Depois basta multiplicar por -J, pois a energia é por definição -J * ∑_pares_vizinhos - h * ∑_todos

        => ΔE = 2 * J * s * sum(neighborSpins)
        """

        s             = self.spins[x,y]
        neighborSpins = [self.spins[x_n,y_n] for x_n,y_n in self.neighbors[x,y]]

        if H == 0:
            return 2 * J * s * np.sum(neighborSpins) # Só para não termos de calcular np.sum(self.spins)
        return     2 * s * (J * np.sum(neighborSpins) + H)

    # Não usado em iter_monte_carlo, mas pode ser útil para outras simulações
    def deltaMag(self, x: int, y:int) -> int:

        """
        ## ΔM ao inverter o spin da célula (x, y) numa grid de largura L e altura igual

        Nota: A magnetização é mais simples, sendo dada por -J * ∑_todos

        A variação de ∑_todos é de -2s, em que s é o spin central (antes era 1, passa a ser -1, variação -2, e vice-versa). Multiplicando por -J,

        ΔM = 2 * J * self.spins[x,y]
        """

        return 2 * J * self.spins[x,y]

    def checkDeltaE(self, deltaE:float, u:float) -> bool: # u é pré-calculado

        """
        ## Para dado ΔE, verifica a condição de aceitação do novo sistema, que é a seguinte:

        e^(-ΔE/k_bT) < u ⇔ ΔE > k_b T ln(u)

        Notas:

        Para -ΔE/k_bT << 0, e^(-ΔE/k_bT) ~ 0, e problemas de precisão de floats podem levar a problemas (o logaritmo não tem este problema)

        Para u = 0, não ∃ ln(u) (o exponencial não tem este problema)
        """

        return deltaE < 0 or np.exp(deltaE * self.ultraConstant) > u

    def record(self):
        """
        *Grava* o estado atual do sistema.
        """

        self.energies.append(self.systemEnergy)              # Adicionar à contagem
        self.magnetizations.append(self.systemMagnetization) # Adicionar à contagem

        return

    def changeSpin(self, x: int, y:int, dE:float) -> float:

        """
        Inverte o spin na célula (x, y), e retorna a variação resultante de energia do sistema

        Nota: dE já está pré-calculada em iter_monte_carlo
        """

        dM = fastDeltaMag(x, y, self.spins)     # Variação de magnetização após inverter o spin em (x, y)
        self.systemEnergy += dE                 # Cálculo da energia do sistema novo somando a variação
        self.systemMagnetization += dM          # Cálculo da magnetização do sistema novo somando a variação

        self.spins[x, y] *= -1

        if DEBUG_MODE:
            i = 0
            if self.systemEnergy != self.__calcSystemEnergy():               i += 1 # ISTO É LENTO, DESLIGAR QUANDO NÃO SE QUER USAR
            if self.systemMagnetization != self.__calcSystemMagnetization(): i -= 2 # ISTO É LENTO, DESLIGAR QUANDO NÃO SE QUER USAR
            if i != 0: raise SimError(i)                                            # Certo: 0, Energia errada: 1, Mag errada: -2, Ambos errados: -1

        return dE                               # Retorna-se deltaE para uso posterior

    def calc_ener_spin(self, x, y):


        s = self.spins[x,y]

        #Contribuição energética local das interações do spin (x,y) com os seus vizinhos. -J para spins iguais, +J spins simétricos.
        neighbor_spins = [self.spins[nx, ny] for nx, ny in self.neighbors[x, y]]
        neighbor_interaction_energy = -J * s * np.sum(neighbor_spins)

        #Contribuição energética do campo externo.
        exterior_field_energy = -H * s

        #Contribuiçao energética local.
        energy =  neighbor_interaction_energy + exterior_field_energy

        return energy

    def calc_ener( self ):
        # calcular a energia por spin do sistema
        # e = E / L^2

        return self.energy / (self.L**2)

    def calc_mag( self ):
        # calcular a magnetização por spin do sistema
        # m = M / L^2
        # ...

        return self.magnetization / (self.L**2)

    def iter_monte_carlo( self, n_iter ):
        randomPointsX = self.rng.integers(low = 0, high = self.L, size = n_iter) # Pré-calcula pontos aleatórios para x, y e u,
        randomPointsY = self.rng.integers(low = 0, high = self.L, size = n_iter) # porque é mais rápido.
        uList         = self.rng.uniform(low = 0, high = 1, size = n_iter)

        # iterar com o método de Metropolis Hastings
        for i in tqdm( range(n_iter), desc=f"L={self.L:6d}, T={self.T:8f}" ):
            x, y, u = randomPointsX[i], randomPointsY[i], uList[i]               # Escolher uma posição aleatória nos spins...
            deltaE = fastDeltaE(x, y, self.spins, self.neighbors)                # ... e ver o deltaE
            if self.checkDeltaE(deltaE, u):
                self.changeSpin(x, y, deltaE)
            self.record()

            if DEBUG_MODE: self.deltaE_difs.append(self.systemEnergy - self.__calcSystemEnergy())

    @property
    def energy(self):
        # usa para aceder ao array com as energias

        return np.array(self.energies)

    @property
    def magnetization(self):
        # usa para aceder ao array com as magnetiza¸c~oes

        return np.array(self.magnetizations)
