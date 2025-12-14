from sympy import diff, Matrix, symbols, Function
import matplotlib.pyplot as plt

from .manifold import Manifold, sp, np


class Submanifold(Manifold):
    def __init__(
            self, 
            ambient_manifold: Manifold, 
            sub_coords: list[symbols], 
            embedding: list[Function]
    ):
        """
        Inizializza una sottovarietà.
        :param ambient_manifold: Istanza della classe Manifold della varietà originale.
        :param sub_coords: Coordinate simboliche della sottovarietà.
        :param embedding: Funzione di immersione che esprime le coordinate globali in termini di quelle della sottovarietà.
        """
        self.ambient_manifold = ambient_manifold
        self.sub_coords = sub_coords
        self.embedding = embedding  # Lista di espressioni simboliche: x_i = f(sub_coords)
        self.dimension = len(sub_coords)

        self.embedding_jacobian = None
        self.induced_metric = None
        self.normal_field = None
        self.second_fundamental_form = None
        self.mean_curvature = None

        self.metric = None  # questo serve per gestire bene l'ereditarietà di certi metodi di Manifold
        self.coords = self.sub_coords  # come ad esempio self.get_christoffels_symbols()

    def get_embedding_jacobian(self):
        self.embedding_jacobian = Matrix([
            [diff(f, coord) for coord in self.sub_coords]
            for f in self.embedding
        ])

        return self.embedding_jacobian

    def get_induced_metric(self):
        """
        Calcola la metrica indotta sulla sottovarietà.
        :return: Matrice simbolica della metrica indotta.
        """
        g = self.ambient_manifold.metric  # Metrica della varietà ambiente
        self.get_embedding_jacobian()  # Jacobiano dell'immersione

        # Metrica indotta: G_ab = (Jacobian)^T * g * (Jacobian)
        self.induced_metric = sp.simplify(self.embedding_jacobian.T * g * self.embedding_jacobian)
        self.metric = self.induced_metric  # serve per poterci agire con metodi di Manifold, e.g. get_christoffel_symbols()
        # self.coords = self.sub_coords

        return self.induced_metric

    def get_normal_field(self):
        """
        Calcola il campo normale della submanifold nell'ambiente.
        :return: Lista di vettori normali simbolici.
        """
        jacobian = self.get_embedding_jacobian()
        ambient_metric = self.ambient_manifold.metric

        tangent_vectors = [jacobian[:, i] for i in range(self.dimension)]

        n = self.ambient_manifold.dimension  # = len(self.embedding)
        k = self.dimension

        comps = []
        normal_vectors = []
        for i in range(n):
            comps.append(f'n{i + 1}')
            normal_vectors.append(sp.symbols(comps[i]))

        # normal_vectors = [sp.symbols(f'n{i + 1}') for i in range(d)]  # Lista di simboli normali
        # initial_symbols = [sp.symbols(f'n{i+1}') for i in range(d)] #ci serve per dopo nel caso in cui n-k>1

        equations = []
        # Condizioni di ortogonalità rispetto alla metrica
        for tangent in tangent_vectors:
            eq = sum(ambient_metric[i, j] * tangent[i] * normal_vectors[j]
                     for i in range(n) for j in range(n))
            equations.append(eq)

        # Normalizzazione: g(N, N) = 1
        norm_eq = sum(ambient_metric[i, j] * normal_vectors[i] * normal_vectors[j]
                      for i in range(n) for j in range(n)) - 1
        equations.append(norm_eq)

        # Risolve il sistema e seleziona il verso giusto
        solutions = sp.solve(equations, normal_vectors)
        # self.compute_scalar_curvature()
        # if self.scalar_curvature >= 0:
        #     self.normal_field = sp.Matrix([solutions[0]])
        # else:
        #      self.normal_field = sp.Matrix([solutions[1]])

        # gestione del caso n-k>1: mancante

        self.normal_field = sp.Matrix([solutions[0]])
        self.normal_field = self.normal_field.subs(sp.I, 1)  # normalizza a reali eventuali vettori complessi
        # questo punto è poco chiaro, non dovrebbe succedere

        # gestione del caso di codimensione > 1
        if n - k >= 2:
            vector_function = sp.Lambda(comps[-1], self.normal_field)
            self.normal_field = vector_function(
                1)  # questo in realtà gestisce solo i casi con vincoli "sferici", come Sn e Hn

        return sp.simplify(self.normal_field.T)




    def get_IInd_fundamental_form(self):
        """
        Calcola la seconda forma fondamentale per la sottovarietà in un ambiente con connessione in generale non piatta.
        :return: Matrice simbolica della seconda forma fondamentale.
        """
        self.get_embedding_jacobian()
        self.ambient_manifold.get_christoffel_symbols()
        Gamma = self.ambient_manifold.christoffel_symbols
        self.get_normal_field()

        II = sp.zeros(self.dimension, self.dimension)
        tangent_vectors = [self.embedding_jacobian[:, i] for i in range(self.dimension)]
        coords = self.sub_coords

        num_vectors = len(tangent_vectors)  # è la dimensione dell'immagine dell'embedding
        num_coords = len(coords)

        # Matrice di derivate, dove ogni elemento è un vettore (colonna)
        derivative_matrix = [[None for _ in range(num_vectors)] for _ in range(num_coords)]

        # Calcolo delle derivate dirette dei vettori tangenti
        for j, tangent_vector in enumerate(tangent_vectors):  # Itera sui vettori tangenti
            for i, coord in enumerate(coords):  # Itera sulle coordinate
                # Calcola la derivata del j-esimo vettore tangente rispetto alla i-esima coordinata
                derivative_matrix[i][j] = tangent_vector.diff(coord)

        # Correzione della connessione
        for i in range(self.dimension):  # indice di derivazione
            for j in range(self.dimension):  # Indice del vettore tangente da derivare
                # Inizializziamo la derivata covariante
                nabla_XY = derivative_matrix[i][j]
                # Inizializzo la derivata covariante come la derivata diretta precedentemente costruita

                # Aggiungo la correzione dei Christoffel
                christoffel_correction = sp.zeros(len(tangent_vectors[0]), 1)  # Vettore colonna
                for k in range(len(tangent_vectors[0])):  # Componente del vettore
                    for m in range(len(tangent_vectors[0])):  # Somma sui vettori tangenti
                        christoffel_correction[k] += Gamma[k][i, m] * tangent_vectors[j][m]

                # Aggiorna la derivata covariante con la correzione
                nabla_XY += christoffel_correction

                # Calcola la proiezione su normal_field per la seconda forma fondamentale
                # II[i, j] = self.normal_field.dot(nabla_XY)
                II[i, j] = self.ambient_manifold.inner_product(nabla_XY, self.normal_field)

        # Salva e restituisce la seconda forma fondamentale
        self.second_fundamental_form = sp.simplify(II)
        return self.second_fundamental_form

    def get_mean_curvatureII(self):
        """
        Calcola la curvatura media della varietà immersa.
        :param: Normal vector field in forma di vettore sympy
        :return: Scalare in forma di sympy function o costante
                Traccia della matrice II
        """
        self.get_IInd_fundamental_form()
        self.get_induced_metric()

        I = self.induced_metric.inv()
        II = self.second_fundamental_form
        # H = 0
        # for a in range(self.dimension):
        #    H += I[a, a] * II[a, a]
        H = sum(
            I[a, a] * II[a, a] for a in range(self.dimension)
        )
        self.mean_curvature = sp.simplify(H)
        return self.mean_curvature

    def is_minimal(self):
        return self.mean_curvature == 0

    def is_totally_geodesic(self):
        n = self.dimension
        self.get_IInd_fundamental_form()
        return self.second_fundamental_form == sp.Matrix.zeros(n, n)

    # di seguito dei doppioni con inserimento manuale del normal vector field
    def get_second_fundamental_form(self, normal_field):
        """
        Calcola la seconda forma fondamentale per la sottovarietà in un ambiente con connessione in generale non piatta.
        :param: normal_field: Campo normale in forma di vettore SymPy.
        :return: Matrice simbolica della seconda forma fondamentale.
        """
        self.get_embedding_jacobian()
        self.ambient_manifold.get_christoffel_symbols()
        Gamma = self.ambient_manifold.christoffel_symbols

        II = sp.zeros(self.dimension, self.dimension)
        tangent_vectors = [self.embedding_jacobian[:, i] for i in range(self.dimension)]
        coords = self.sub_coords

        num_vectors = len(tangent_vectors)  # è la dimensione dell'immagine dell'embedding
        num_coords = len(coords)

        # Matrice di derivate, dove ogni elemento è un vettore (colonna)
        derivative_matrix = [[None for _ in range(num_vectors)] for _ in range(num_coords)]

        # Calcolo delle derivate dirette dei vettori tangenti
        for j, tangent_vector in enumerate(tangent_vectors):  # Itera sui vettori tangenti
            for i, coord in enumerate(coords):  # Itera sulle coordinate
                # Calcola la derivata del j-esimo vettore tangente rispetto alla i-esima coordinata
                derivative_matrix[i][j] = tangent_vector.diff(coord)

        # Correzione della connessione
        for i in range(self.dimension):  # indice di derivazione
            for j in range(self.dimension):  # Indice del vettore tangente da derivare
                # Inizializziamo la derivata covariante
                nabla_XY = derivative_matrix[i][j]
                # Inizializzo la derivata covariante come la derivata diretta precedentemente costruita

                # Aggiungo la correzione dei Christoffel
                christoffel_correction = sp.zeros(len(tangent_vectors[0]), 1)  # Vettore colonna
                for k in range(len(tangent_vectors[0])):  # Componente del vettore
                    for m in range(len(tangent_vectors[0])):  # Somma sui vettori tangenti
                        christoffel_correction[k] += Gamma[k][i, m] * tangent_vectors[j][m]

                # Aggiorna la derivata covariante con la correzione
                nabla_XY += christoffel_correction

                # Calcola la proiezione su normal_field per la seconda forma fondamentale
                II[i, j] = normal_field.dot(nabla_XY)

        # Salva e restituisci la seconda forma fondamentale
        self.second_fundamental_form = sp.simplify(II)
        return self.second_fundamental_form

    def get_mean_curvature(self, normal_field):
        """
        Calcola la curvatura media della varietà immersa.
        :param: Normal vector field in forma di vettore sympy
        :return: Scalare in forma di sympy function o costante
                Traccia della matrice II
        """
        self.get_second_fundamental_form(normal_field)
        self.get_induced_metric()

        I = self.induced_metric.inv()
        II = self.second_fundamental_form
        H = 0
        for a in range(self.dimension):
            for b in range(self.dimension):
                if a == b:
                    H += I[a, b] * II[a, b]

        self.mean_curvature = sp.simplify(H)
        return self.mean_curvature

    def plot_surface(self, domain, fig_title='Surface'):
        """:param domain: it's a list made of 2 tuples giving the intervals of the variables parametrizing the surface"""

        coords = self.sub_coords
        x, y, z = self.embedding[0], self.embedding[1], self.embedding[2]

        flg_null = None
        for i, c in enumerate([x, y, z]):
            if c == 0:
                flg_null = i

        func = [sp.lambdify((coords[0], coords[1]), coord, 'numpy') for coord in [x, y, z]]
        # questo mi produce e.g. [x(u,v), y(u,v), z(u,v)]

        # Creiamo la meshgrid per le coordinate
        a1, b1, a2, b2 = domain[0][0], domain[0][1], domain[1][0], domain[1][1]
        u = np.linspace(a1, b1, 100)
        v = np.linspace(a2, b2, 100)
        U, V = np.meshgrid(u, v)

        # Valutiamo le coordinate cartesiane
        Func = [c(U, V) for c in func]

        if flg_null is not None:  # gestisce i casi con una coordinata nulla
            if flg_null == 0:
                Func[flg_null] = np.zeros_like(Func[flg_null + 1])
            else:  # elif flg_null == 1 or flg_null == 2:
                Func[flg_null] = np.zeros_like(Func[flg_null - 1])

        # Plot
        fig = plt.figure(figsize=(6, 6))

        ax = fig.add_subplot(111, projection='3d')
        ax.set_title(fig_title, fontsize=14)

        if flg_null is not None:
            ax.plot_surface(Func[0], Func[1], Func[2], color='c', cmap='cividis', edgecolor='none', alpha=0.9,
                            shade=True)
        else:
            ax.plot_surface(Func[0], Func[1], Func[2], color='c', cmap='viridis', edgecolor='none', alpha=0.9, shade=True)

        # Etichette degli assi
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")

        # Rimuoviamo i gridlines per un aspetto più pulito
        # ax.grid(False)
        # plt.axis('off')

        # Aggiunge una barra del colore
        fig.colorbar(ax.plot_surface(Func[0], Func[1], Func[2], color='c', cmap='viridis', edgecolor='none', alpha=0.9,
                                     shade=True))

        return plt.show()

    def plot_geodesics_on_surface(self, domain, geodesics, fig_title='Geodesics'):
        """
        Plotta la superficie immersa in R3 e le geodetiche sopra di essa.

        :param domain: Lista con due tuple che danno gli intervalli delle coordinate parametrizzanti la submanifold.
        :param geodesics: Lista di soluzioni numeriche delle equazioni geodetiche.
        """

        coords = self.sub_coords
        x, y, z = self.embedding[0], self.embedding[1], self.embedding[2]

        flg_null = None
        for i, c in enumerate([x, y, z]):
            if c == 0:
                flg_null = i

        # Funzioni di embedding lambda per valutare le coordinate 3D
        func = [sp.lambdify((coords[0], coords[1]), coord, 'numpy') for coord in [x, y, z]]

        # Creiamo la meshgrid per la superficie
        a1, b1, a2, b2 = domain[0][0], domain[0][1], domain[1][0], domain[1][1]
        u = np.linspace(a1, b1, 100)
        v = np.linspace(a2, b2, 100)
        U, V = np.meshgrid(u, v)

        # Valutiamo l'immersione nello spazio 3D
        Func = [c(U, V) for c in func]

        if flg_null is not None:  # gestisce i casi con una coordinata nulla
            if flg_null == 0:
                Func[flg_null] = np.zeros_like(Func[flg_null + 1])
            else:  # elif flg_null == 1 or flg_null == 2:
                Func[flg_null] = np.zeros_like(Func[flg_null - 1])

        # Plot
        fig = plt.figure(figsize=(6, 6))

        ax = fig.add_subplot(111, projection='3d')
        ax.set_title(fig_title, fontsize=14)

        if flg_null is not None:
            ax.plot_surface(Func[0], Func[1], Func[2], color='c', cmap='cividis', edgecolor='none', alpha=0.8,
                            shade=True)
        else:
            ax.plot_surface(Func[0], Func[1], Func[2], color='c', cmap='viridis', edgecolor='none', alpha=0.6, shade=True)

        # Plot delle geodetiche con più spessore
        for sol in geodesics:
            u_vals, v_vals = sol.y[0], sol.y[1]  # Coordinate sulla submanifold

            # Mappiamo le coordinate della geodetica nell'immersione 3D
            x_vals = func[0](u_vals, v_vals)
            y_vals = func[1](u_vals, v_vals)
            z_vals = func[2](u_vals, v_vals)

            ax.plot(x_vals, y_vals, z_vals, color='r', linewidth=3)

        # Etichette degli assi
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        # ax.view_init(elev=30, azim=45) possiamo ruotare il plot

        # Aggiunge una barra del colore
        fig.colorbar(ax.plot_surface(Func[0], Func[1], Func[2], color='c', cmap='viridis', edgecolor='none', alpha=0.9,
                                     shade=True))

        return plt.show()

    def get_geometrics_sub(self):
        """Compute the main geometric objects of a (sub)manifold"""
        self.get_induced_metric()
        self.get_einstein_tensor()
        self.get_kretschmann_scalar()
        self.get_sectional_curvature_matrix()
        self.get_geodesic_equations()
        self.get_mean_curvatureII()


    def IInd_norm_squared(self):
        """Compute the squared norm ∣A∣2=h^ij h_ij of the IInd form h"""
        n = self.dimension
        h = self.get_IInd_fundamental_form()
        g = self.get_induced_metric()
        g_inv = g.inv()

        h_up = sp.Matrix.zeros(n, n)
        for i in range(n):
            for j in range(n):
                h_up[i, j] = sum(
                    g_inv[i, k] * g_inv[j, l] * h[k, l]
                    for k in range(n) for l in range(n)
                )
        A_squared = sum(h_up[i, j] * h[i, j] for i in range(n) for j in range(n))
        return sp.simplify(A_squared)


    def get_orth_frame(self, embedding):
        """Fornisce un frame ortonormale per una hypersurface
            :param embedding: è una lista con le componenti dell'immagine dell'embedding"""
        n = self.dimension+1
        Rn = Manifold(sp.Matrix.eye(n), [sp.symbols(f'x_{j}') for j in range(n)])

        hypersurface = Submanifold(Rn, self.sub_coords, embedding)
        hypersurface.get_normal_field()

        frame = [hypersurface.embedding_jacobian[:, j] for j in range(self.dimension)]
        frame.append(hypersurface.normal_field.T)
        return frame

    def Jacobi_operator(self, f):
        """Compute the Jacobi operator J(f)=Δf+(∣A∣2+Ric(N,N))f on the Submanifold
          applied to an input function
        :param f: sympy function like sp.Function('f')(sp.symbols('u'), sp.symbols('v'))
        """

        n, k = self.ambient_manifold.dimension, self.dimension
        if n - k > 1:
            """dobbiamo definire una manifold "intermedia" che fa da vero ambiente per i casi tipo S2
                dove abbiamo (theta,phi) --> (x,y,z,0); infatti in questo caso verrebbe letto che S2 è
                immerso in R4 invece che in S3 e il Ricci risulterebbe nullo e non quello di S3.
            """

            # prima gestiamo il campo normale N, che dovrà essere tale che (∂1,...,∂k,N) sia un frame di R^m, m=k+1=n-1
            embedding = self.embedding[:-1]
            frame = self.get_orth_frame(embedding)

            N = frame[-1]

            if any(isinstance(expr, sp.Basic) and expr.has(sp.sinh, sp.cosh) for expr in
                   self.embedding):  # caso hyperbolic Hm, m=k+1

                Ric = Hyp(k+1).ricci_tensor

            elif (all(not (isinstance(expr, sp.Basic) and expr.has(sp.sinh, sp.cosh)) for expr in self.embedding)
                  and any(isinstance(expr, sp.Basic) and expr.has(sp.sin, sp.cos) for expr in
                          self.embedding)):  # caso spheric Sm, m=k+1

                Ric = Sphere(k+1).ricci_tensor

        else:
            Ric = self.ambient_manifold.get_ricci_tensor()
            N = self.get_normal_field()

        A_squared = self.IInd_norm_squared()  # questo valorizza anche "self.induced_metric" e "self.normal_field"
        Delta_f = self.laplacian(f)
        Ric_N_N = N.T * (Ric * N)  # è ancora una MutableDenseNDimArray della forma [scalar]

        sp.pprint(Ric_N_N) #DEBUG

        return sp.simplify(Delta_f + A_squared * f + Ric_N_N[0] * f)
    


class Sphere(Submanifold):
    def __init__(self, dimension):
        """Classe per le sfere Sn"""
        self.dimension = dimension
        self.coords = [sp.symbols(f'θ{j+1}') for j in range(self.dimension)]
        self.sub_coords = self.coords  # per gestire l'ereditarietà
        self.ambient_manifold = Manifold(sp.Matrix.eye(self.dimension + 1),
                                         [sp.symbols(f'x{j}') for j in range(self.dimension + 1)])
        self.embedding = self.get_embedding()
        self.metric = self.get_induced_metric()

        self.get_geometrics()

    def get_embedding(self):
        """
        Restituisce l'embedding della sfera Sn in R^(n+1).
        """
        embedding_coords = []

        # Costruzione delle coordinate nell'iperspazio
        for i in range(self.dimension + 1):
            coord = 1  # Fattore iniziale

            # Moltiplica i seni delle coordinate precedenti
            for j in range(i):
                coord *= sp.sin(self.coords[j])

            # L'ultimo angolo si usa per un coseno, altrimenti si usa un seno
            if i < self.dimension:
                coord *= sp.cos(self.coords[i])

            embedding_coords.append(coord)

        return embedding_coords


class Hyp(Submanifold):
    def __init__(self, dimension):
        """Classe per gli iperboloidi Hn"""
        self.dimension = dimension
        self.coords = [sp.symbols('s')] + [sp.symbols(f'θ{j+1}') for j in range(self.dimension - 1)]
        self.sub_coords = self.coords  # per gestire l'ereditarietà

        g_Sn_1 = Sphere(self.dimension - 1).metric
        g = sp.Matrix.eye(self.dimension)
        g[1:, 1:] = sp.sinh(self.coords[0]) ** 2 * g_Sn_1

        self.metric = g
        self.get_geometrics()




class Eucl(Submanifold):
    def __init__(self, dimension):
        """Classe per gli gli spazi Euclidei Rn"""
        self.dimension = dimension
        self.coords = [sp.symbols(f'x{j+1}') for j in range(self.dimension)]
        self.sub_coords = self.coords  # per gestire l'ereditarietà

        self.metric = sp.Matrix.eye(self.dimension)
        self.get_geometrics()

        # polar coordinates
        r = sp.symbols('r')
        g = sp.Matrix.eye(self.dimension)
        g[1:, 1:] = r**2*Sphere(self.dimension-1).metric

        self.polar = Manifold(g, [r]+[sp.symbols(f'θ{j+1}') for j in range(self.dimension-1)])
        self.polar.get_geometrics()




class Minkowski(Submanifold):
    def __init__(self, neg, plus):
        """Classe per gli spazi di Minkowski R^(1,n)"""
        self.neg = neg
        self.plus = plus
        self.dimension = neg + plus
        self.coords = [sp.symbols(f'x{j}') for j in range(self.dimension)]
        self.sub_coords = self.coords  # per gestire l'ereditarietà

        g = sp.Matrix.eye(self.dimension)
        g[:self.neg, :self.neg] = -sp.Matrix.eye(self.neg)

        self.metric = g
        self.get_geometrics()



class Schwarzschild(Submanifold):
     def __init__(self, dimension):
        """Classe per le foglie Schwarzschild spacelike"""

        self.dimension = dimension

        r, G, m = sp.symbols('r G m')
        u = sp.sqrt(1- 2*G*m*r**(2-self.dimension))

        self.coords = [r] + [sp.symbols(f'θ{j+1}') for j in range(self.dimension-1)]
        self.sub_coords = self.coords  # per gestire l'ereditarietà

        g = sp.Matrix.eye(self.dimension)
        g[0, 0] = 1/u**2
        g[1:, 1:] = r**2*Sphere(self.dimension-1).metric

        self.metric = g

        self.get_geometrics()

        g_spacetime = sp.Matrix.eye(self.dimension+1)
        g_spacetime[1:, 1:] = self.metric
        g_spacetime[0, 0] = -u**2

        self.spacetime_coords = [sp.symbols('t')] + self.coords
        self.spacetime = Manifold(g_spacetime, self.spacetime_coords)
        self.spacetime.get_geometrics()

        horizon_embedding = [0, 2*sp.symbols('G')*sp.symbols('m')] + self.coords[1:]
        self.horizon = Submanifold(self.spacetime, self.coords[1:], horizon_embedding)
        self.horizon.get_geometrics_sub()
