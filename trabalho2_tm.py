from typing import Dict, List, Optional, Tuple


def parse_graph(tape_string: str):
    """
    Interpreta a entrada codificada na fita no formato:

        n#u1,v1;u2,v2;...;um,vm

    Nesta funcao, a fita ainda contem apenas a entrada bruta.
    O objetivo e separar:
    - n: quantidade de vertices;
    - edges: lista de arestas.

    Se a codificacao estiver invalida, retorna None.
    """

    try:
        # Separa a quantidade de vertices da lista de arestas.
        parts = tape_string.split("#")
        if len(parts) != 2:
            return None

        # A parte da esquerda precisa ser um numero inteiro.
        n = int(parts[0])

        edges = []

        # Grafo sem arestas tambem e valido.
        if parts[1] == "":
            return n, edges

        # Cada aresta e escrita como "u,v" e separada por ";".
        for edge in parts[1].split(";"):
            vertices = edge.split(",")
            if len(vertices) != 2:
                return None

            # Converte os dois extremos da aresta para inteiros.
            u = int(vertices[0])
            v = int(vertices[1])
            edges.append((u, v))

        return n, edges
    except Exception:
        return None


def build_graph(n: int, edges: List[Tuple[int, int]]):
    """
    Constroi uma lista de adjacencia apenas para validar a entrada.

    A MT nao usa essa estrutura como memoria principal da decisao.
    Ela serve para conferir se os vertices informados realmente existem.
    """

    # O numero de vertices precisa ser positivo.
    if n <= 0:
        return None

    # Cria uma lista vazia de vizinhos para cada vertice.
    graph = {vertex: [] for vertex in range(1, n + 1)}

    for u, v in edges:
        # Cada extremidade precisa estar dentro do intervalo valido.
        if u < 1 or u > n:
            return None
        if v < 1 or v > n:
            return None

        # O grafo e nao direcionado, entao a aresta entra nos dois lados.
        graph[u].append(v)
        graph[v].append(u)

    return graph


class TuringMachine:
    """
    Simulacao de uma Maquina de Turing para decidir se um grafo e bipartido.

    A ideia e seguir o modelo de MT:
    - a entrada fica escrita na fita;
    - a cabeca de leitura/escrita aponta para uma celula de cada vez;
    - o estado atual controla o que a maquina faz;
    - a fita tambem guarda a memoria de trabalho, como as cores dos vertices.

    Esta versao nao usa BFS com fila em memoria principal.
    Em vez disso, a MT faz varreduras sucessivas sobre a fita e sobre a tabela
    de cores, movimentando a cabeca passo a passo.
    """

    def __init__(self, tape):
        # Simbolo branco da fita.
        self.blank = "â–¡"

        # A fita e representada por uma lista de simbolos.
        self.tape = list(tape)

        # Se a entrada vier vazia, garante pelo menos uma celula branca.
        if not self.tape:
            self.tape.append(self.blank)

        # Cabeca de leitura/escrita: este e o ponteiro da MT sobre a fita.
        self.head = 0

        # Estado inicial da maquina.
        self.state = "q0"

        # Contador de passos da simulacao.
        self.steps = 0

        # Estruturas auxiliares para interpretar a entrada ja lida.
        self.graph = None
        self.edges = []
        self.vertices = []

        # Mapeia cada vertice para a posicao da sua celula de cor na fita.
        # Assim, a cabeca consegue voltar exatamente ao ponto correto.
        self.color_positions: Dict[int, int] = {}

        # Ponteiro logico para o proximo vertice a ser examinado.
        self.vertex_cursor = 0

        # Vertice ativo no momento.
        self.active_vertex: Optional[int] = None

        # Cor atual do vertice ativo.
        self.active_color: Optional[str] = None

        # Indice da aresta sendo lida na varredura atual.
        self.edge_index = 0

    # --------------------------------------------------
    # Operacoes sobre a fita
    # --------------------------------------------------

    def read_symbol(self):
        """
        Le o simbolo que esta na celula apontada pela cabeca.

        Se a cabeca sair dos limites, a fita e expandida com simbolos brancos.
        Isso imita a fita infinita de uma Maquina de Turing.
        """

        # Se a cabeca andar para a esquerda alem do inicio, criamos espaco.
        while self.head < 0:
            self.tape.insert(0, self.blank)
            self.head += 1

        # Se a cabeca ultrapassar o fim, estendemos a fita a direita.
        while self.head >= len(self.tape):
            self.tape.append(self.blank)

        return self.tape[self.head]

    def write_symbol(self, symbol):
        """
        Escreve um simbolo na celula apontada pela cabeca.

        Em uma MT, leitura e escrita acontecem no mesmo ponto da fita.
        """

        # Garante que a posicao atual exista antes de escrever.
        while self.head < 0:
            self.tape.insert(0, self.blank)
            self.head += 1

        while self.head >= len(self.tape):
            self.tape.append(self.blank)

        self.tape[self.head] = symbol

    def move_head(self, direction):
        """
        Move a cabeca da MT:
        - R: uma celula para a direita;
        - L: uma celula para a esquerda;
        - S: permanece parada.
        """

        if direction == "R":
            self.head += 1
        elif direction == "L":
            self.head -= 1
        elif direction == "S":
            pass

        # Mantem a fita sempre expansivel.
        if self.head < 0:
            self.tape.insert(0, self.blank)
            self.head = 0
        elif self.head >= len(self.tape):
            self.tape.append(self.blank)

    def _set_head(self, target):
        """
        Posiciona a cabeca em um indice especifico da fita.

        Isto e importante conceitualmente: uma MT nao acessa memoria
        aleatoriamente. Ela precisa mover a cabeca celula por celula.
        """

        while self.head < target:
            self.move_head("R")
        while self.head > target:
            self.move_head("L")

    def _base_color(self, symbol):
        """
        Normaliza a cor lida na fita.

        As letras minusculas indicam vertices ja processados,
        entao convertemos para a cor base correspondente.
        """

        if symbol in {"R", "r"}:
            return "R"
        if symbol in {"G", "g"}:
            return "G"
        return symbol

    def _opposite_color(self, symbol):
        """Retorna a cor oposta da cor recebida."""
        return "G" if symbol == "R" else "R"

    def _color_position(self, vertex):
        """Recupera a posicao da celula de cor do vertice informado."""
        return self.color_positions[vertex]

    def _read_vertex_color(self, vertex):
        """
        Move a cabeca ate a celula do vertice e le sua cor.

        Aqui a fita faz o papel da memoria da MT, e a cabeca e o ponteiro
        que vai ate a posicao correta para consultar o estado do vertice.
        """

        position = self._color_position(vertex)
        self._set_head(position)
        return self.read_symbol()

    def _write_vertex_color(self, vertex, symbol):
        """
        Move a cabeca ate a celula do vertice e escreve a nova cor.

        Esta operacao simula a atualizacao da memoria de trabalho da MT.
        """

        position = self._color_position(vertex)
        self._set_head(position)
        self.write_symbol(symbol)

    def _initialize_work_area(self, n):
        """
        Monta uma area auxiliar na fita para armazenar as cores.

        A estrutura fica assim, de forma conceitual:

        entrada original | C: 1U;2U;3U;... |

        Onde:
        - U = vertice sem cor;
        - R = uma cor;
        - G = a outra cor;
        - r/g = vertice ja processado.
        """

        # Se a entrada ainda nao foi separada da area de trabalho, adiciona um marcador.
        if self.tape and self.tape[-1] != "|":
            self.tape.append("|")

        # Marca o inicio da regiao de trabalho da MT.
        self.tape.extend(list("C:"))
        self.color_positions = {}

        # Cria uma celula de cor para cada vertice.
        for vertex in range(1, n + 1):
            # Mantem o numero do vertice gravado antes da cor.
            for ch in str(vertex):
                self.tape.append(ch)

            # Esta posicao passa a representar a cor do vertice.
            color_pos = len(self.tape)
            self.tape.append("U")
            self.color_positions[vertex] = color_pos

            # Separador visual entre vertices.
            self.tape.append(";")

        # Marca o fim da area auxiliar.
        self.tape.append("|")

    # --------------------------------------------------
    # Uma transicao da MT
    # --------------------------------------------------

    def step(self):
        """
        Executa uma transicao da Maquina de Turing.

        Cada chamada representa uma pequena mudanca de estado,
        com leitura, escrita e movimentacao da cabeca.
        """

        # Cada transicao consome um passo da simulacao.
        self.steps += 1

        # q0: interpreta a entrada e prepara a area de trabalho.
        if self.state == "q0":
            # Lemos a fita inteira como a entrada escrita pelo usuario.
            tape_string = "".join(self.tape).replace(self.blank, "")
            graph_data = parse_graph(tape_string)

            # Entrada invalida leva a rejeicao imediata.
            if graph_data is None:
                self.state = "qReject"
                return False

            n, edges = graph_data
            graph = build_graph(n, edges)

            # Se houver vertices fora do intervalo, a MT rejeita.
            if graph is None:
                self.state = "qReject"
                return False

            # Guarda a estrutura do grafo apenas como apoio da simulacao.
            self.graph = graph
            self.edges = edges
            self.vertices = list(range(1, n + 1))

            # Cria a tabela de cores diretamente na fita.
            self._initialize_work_area(n)

            # Reinicia os ponteiros logicos da simulacao.
            self.vertex_cursor = 0
            self.active_vertex = None
            self.active_color = None
            self.edge_index = 0

            # Proximo estado: procurar um vertice para processar.
            self.state = "qSelectVertex"
            return True

        # qSelectVertex: procura o proximo vertice ainda nao processado.
        if self.state == "qSelectVertex":
            # Varre os vertices na ordem 1..n.
            while self.vertex_cursor < len(self.vertices):
                vertex = self.vertices[self.vertex_cursor]

                # Posiciona a cabeca na celula correspondente ao vertice.
                symbol = self._read_vertex_color(vertex)

                # Se ainda estiver sem cor, a MT escolhe uma cor inicial.
                if symbol == "U":
                    self._write_vertex_color(vertex, "R")
                    symbol = "R"

                # Se o vertice ja tem cor, ele vira o vertice ativo da rodada.
                if symbol in {"R", "G"}:
                    self.active_vertex = vertex
                    self.active_color = symbol
                    self.edge_index = 0
                    self.state = "qScanEdges"
                    return True

                # Caso inesperado, apenas avanca para o proximo.
                self.vertex_cursor += 1

            # Se nenhum vertice restar para processar, a MT aceita.
            self.state = "qAccept"
            return False

        # qScanEdges: varre todas as arestas procurando conflitos.
        if self.state == "qScanEdges":
            # Se todas as arestas ja foram examinadas, finaliza o vertice atual.
            if self.edge_index >= len(self.edges):
                # Marca o vertice como processado usando minuscula.
                if self.active_vertex is not None and self.active_color is not None:
                    self._write_vertex_color(
                        self.active_vertex,
                        self.active_color.lower(),
                    )

                # Limpa o vertice ativo e prepara a proxima rodada.
                self.active_vertex = None
                self.active_color = None
                self.vertex_cursor += 1
                self.state = "qSelectVertex"
                return True

            # Lemos a proxima aresta da lista interpretada.
            u, v = self.edges[self.edge_index]
            self.edge_index += 1

            # Se a aresta nao toca o vertice ativo, ela e ignorada nesta rodada.
            if self.active_vertex not in (u, v):
                return True

            # Identifica o vizinho do vertice ativo nesta aresta.
            neighbor = v if u == self.active_vertex else u

            # A cabeca vai ate a celula do vizinho para consultar sua cor.
            neighbor_symbol = self._read_vertex_color(neighbor)
            neighbor_color = self._base_color(neighbor_symbol)

            # Se o vizinho ainda nao tem cor, grava a cor oposta na fita.
            if neighbor_symbol == "U":
                self._write_vertex_color(
                    neighbor,
                    self._opposite_color(self.active_color),
                )
                return True

            # Se o vizinho ja tem a mesma cor, existe conflito e a MT rejeita.
            if neighbor_color == self.active_color:
                self.state = "qReject"
                return False

            # Sem conflito, continua a varredura da fita.
            return True

        # Estados terminais apenas interrompem a computacao.
        return False

    # --------------------------------------------------
    # Execucao completa
    # --------------------------------------------------

    def execute(self, step_limit=300000):
        """
        Executa a MT ate aceitar, rejeitar ou atingir um limite de passos.

        O limite e uma protecao contra loops causados por erros de entrada
        ou por alguma inconsistencia de implementacao.
        """

        while self.steps < step_limit:
            keep_running = self.step()

            # Se a MT chegou ao estado de aceite, retornamos sucesso.
            if self.state == "qAccept":
                return True, self.steps, "".join(self.tape)

            # Se a MT chegou ao estado de rejeicao, retornamos falha.
            if self.state == "qReject":
                return False, self.steps, "".join(self.tape)

            # Se a transicao indicar parada, encerramos o laço.
            if not keep_running:
                break

        # Se o limite for atingido, tratamos como rejeicao segura.
        return False, self.steps, "".join(self.tape)
