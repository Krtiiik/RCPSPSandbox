# RCPSPSandbox

Cílem tohoto projektu je vytvořit program s následujícími třemi funkcemi:

1. Obohacování standardních [PSBLIB](https://www.om-db.wi.tum.de/psplib/library.html) instancí RCPSP o deadlines, podrozdělování příkazů,
   zavádění kapacit zdrojů. Následně bude převádět data do univerzálního datového formátu (pravděpodobně JSON).
2. Řešení problému ve výše definovaném formátu pomocí IBM CPLEX Optimizer solveru.
3. Sandbox pro testování různých přístupů pro zlepšování existujícího řešení vzhledem k požadovaným kritériím.

Třetí část poskytne prostor k testování přístupů k relaxaci původního problému s cílem zlepšení tardiness kritéria specifikovaných příkazů.
