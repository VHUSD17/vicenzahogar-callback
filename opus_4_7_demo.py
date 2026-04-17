"""
De qué está hecho Claude Opus 4.7
=================================
Un pequeño programa que imprime los "ingredientes" del modelo
y demuestra algunas de sus capacidades en acción.
"""

from dataclasses import dataclass, field
from datetime import date


@dataclass
class Ingrediente:
    nombre: str
    porcentaje: float
    descripcion: str


@dataclass
class Opus47:
    model_id: str = "claude-opus-4-7"
    familia: str = "Claude 4.X"
    fecha_conocimiento: date = date(2026, 1, 1)
    ingredientes: list[Ingrediente] = field(default_factory=lambda: [
        Ingrediente("Razonamiento profundo",  22, "Cadenas de pensamiento largas y coherentes."),
        Ingrediente("Código",                 20, "Lectura, escritura y refactor de proyectos enteros."),
        Ingrediente("Uso de herramientas",    15, "Llamadas en paralelo, MCP, agentes y sub-agentes."),
        Ingrediente("Contexto extendido",     12, "Ventana amplia con compresión automática."),
        Ingrediente("Seguridad y alineación", 10, "Rechazos calibrados, respuestas responsables."),
        Ingrediente("Multilingüe",             8, "Español, inglés y decenas de idiomas más."),
        Ingrediente("Visión",                  7, "Lee imágenes, PDFs y notebooks."),
        Ingrediente("Humor y estilo",          6, "Tono conciso, directo y amable."),
    ])

    def receta(self) -> str:
        total = sum(i.porcentaje for i in self.ingredientes)
        lineas = [f"Receta de {self.model_id}  (familia {self.familia})",
                  "-" * 48]
        for ing in self.ingredientes:
            barra = "#" * int(ing.porcentaje / 2)
            lineas.append(f"{ing.porcentaje:>3}% {barra:<12} {ing.nombre}")
            lineas.append(f"         -> {ing.descripcion}")
        lineas.append("-" * 48)
        lineas.append(f"Total: {total}%  |  cutoff: {self.fecha_conocimiento}")
        return "\n".join(lineas)

    def demo_capacidades(self) -> dict[str, str]:
        return {
            "razonar":     self._fibonacci(10),
            "codear":      self._fizzbuzz(15),
            "traducir":    "Hola -> Hello -> Bonjour -> Ciao -> こんにちは",
            "resumir":     "Opus 4.7 = razonamiento + código + herramientas + cuidado.",
        }

    @staticmethod
    def _fibonacci(n: int) -> str:
        a, b, serie = 0, 1, []
        for _ in range(n):
            serie.append(a)
            a, b = b, a + b
        return " ".join(map(str, serie))

    @staticmethod
    def _fizzbuzz(n: int) -> str:
        out = []
        for i in range(1, n + 1):
            if i % 15 == 0:  out.append("FizzBuzz")
            elif i % 3 == 0: out.append("Fizz")
            elif i % 5 == 0: out.append("Buzz")
            else:            out.append(str(i))
        return " ".join(out)


if __name__ == "__main__":
    opus = Opus47()
    print(opus.receta())
    print()
    print("Demo de capacidades:")
    for habilidad, resultado in opus.demo_capacidades().items():
        print(f"  [{habilidad:8}] {resultado}")
