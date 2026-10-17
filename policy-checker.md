Como manager quiero crear un mecanismo de revisión de políticas de procedimientos de gestión en el contexto de mi trabajo.
Haremos uso del soporte MCP de Vatuta para añadir fuentes de datos sobre las que hacer la recolección de datos sobre los que hacer las validaciones. A modo de ejemplo y para validar la solución SOLO lo haremos sobre una cuenta de Github y de JIRA Free.
Existen dos casos de uso:
1.- Dado una política definida con lenguaje natural, un documento de descripción formal, quiero extraer todas las validaciones y el mecanismo para validarlas una a una (Creación de validaciones)
2.- Dada la lista de validaciones, quiero poder validar todas y cada una de ellas y obtener un listado de violaciones
Vamos a usar un nuevo comando del cliente de vatuta llamado policy. Cubrirá los dos casos de uso anteriores.
Un ejemplo de política puede ser el hecho de que todo pull request de github tenga un título siguiendo una nomenclatura específica que contenga la referencia al ticket de JIRA free que tiene que tener un estado específico. (Verificación de políticas)
Esta política está definida entre otras en un documento descriptivo formal, escrito por un humano con lenguaje natura.

Vatuta (en el primer caso de uso):
1.- Leer el documento de políticas y el prompt de creación de validaciones donde puede extraer posibles parámetros
2.- Identifica y enumera todas las políticas o validaciones (a modo de controles atómicos)
3.- Analiza las fuentes de datos existentes
4.- Crea el mecanismo determinista (en la medida de lo posible) que verifica la política y extrae el listado de violaciones de la política

Vatuta (en el segundo caso de uso):
1.- Ejecuta cada una de las validaciones existentes
2.- Haciendo uso de las fuentes de datos para recolectar información y de los posibles parámetros extraidos del prompt de verificación de políticas
2.- Devuelve el listado de violaciones

Cada **control independiente/atómico de validación** seguirá la metodología RASE:

* R — Requirement: qué debe cumplirse
* A — Applicability: cuándo aplica
* S — Selection: alternativas o selección de casos
* E — Exception: cuándo deja de aplicar
Como por ejemplo el siguiente:

```yaml
id: JIRA-027

source:
  document: Software Development Policy
  section: 4.7.2
  version: 3.2

applicability:
  project: {{project parameter}}
  issue_type: Epic
  criticality: critical

exceptions:
  - emergency_change

requirement:
  type: semantic
  statement: >
    Risks affecting production must be documented.

selection:
  one_of:
    - mechanism: dedicated_risk_field
      evidence:
        - jira.customfield.risk

    - mechanism: description_section
      evidence:
        - jira.description

    - mechanism: jira_comment
      evidence:
        - jira.comments

evidence:
  - jira.description
  - jira.customfield.risk
  - jira.comments

result:
  - compliant
  - non_compliant
  - insufficient_evidence
  - not_applicable
```

Los controles pueden ser de varios tipos:

* Determinista: Código de ejecución determinista comprueba que se cumple la política. O la propia consulta a la fuente devuelve las violaciones.
* Semántico: Una consulta a un LLM (o JEV) hace una evaluación de cumplimiento de política
* Híbrido: Mezcla de ambos
* Human review: Requiere intervención humana para evaluar el cumplimiento (para aquellos casos en los que el control no se puede automatizar)

Priorizando siempre los controles deterministas sobre los semánticos. Y los semánticos sobre los humanos.

El resultado del control puede ser de dos tipos depende de la naturaleza del control:

* Listado de violaciones, si devuelve todos los casos de incumplimiento de la política (ie. tickets de jira sin SP asignados)
* Resultado de cumplimiento de la política (ie. La varianza de velocity entre sprints no puede ser mayor del 20%)
  El resultado puede ser:
  * Compliant: Cumple la política
  * Non compliant: No cumple la política
  * Insufficient evidence: No hay evidencia suficiente para evaluar el cumplimiento
  * Not applicable: No aplica
Este es el pseudocódigo:

```python
if not evidence_available(control):
    return INSUFFICIENT_EVIDENCE

if not applicability.evaluate(data):
    return NOT_APPLICABLE

if requirement.evaluate(data):
    return COMPLIANT

return NON_COMPLIANT
```

La ejecución por control puede llevar la siguiente secuencia:
1.- Extracción de los datos de la fuente: **Source collection**
2.- Agregación/filtrado/proyección de los datos de la fuente para obtener la información relevante. Lo que denominamos **evidencias**: **Evidence acquisition**
3.- **Facts normalization**
3.- **Control evaluation**: Evaluación de la información extraida y comparación con la política, que puede ser con una condición determinista (ej: status != done) o semántica (ej: Descripción tiene que contener "PR"). Esta evaluación incluye ver si en las evidencias se cumplen los criterios de aplicabilidad y los requisitos siguiendo los diferentes selectores.: **Evidence evaluation**
4.- Devolución de resultado, que puede ser un listado de violaciones o un resultado de cumplimiento. **Result evaluation**.

Un control podría depender del resultado de otro control. En ese caso la ejecución del segundo debe poder recuperar como evidencia el resultadol del primero. En ese caso el orden de ejecución de los controles debe tenerlo en cuenta

La evaluación se hace con Common Expression Language (CEL) o JSON schema para el caso determinista y con LLM o modelo tipo JEV de TypeSafe AI para el caso semántico

Otro ejemplo de policy/control:

```yaml
id: JIRA-042

applicability:
  evaluator: cel
  expression: >
    issue.type == "Bug" &&
    issue.priority == "Critical"

evidence:
  required:
    - issue.assignee
    - issue.status
    - issue.transitions
    - issue.approvals

requirement:
  evaluator: cel
  expression: >
    has(issue.assignee) &&
    issue.approvals.exists(
      a,
      a.role == "ProductOwner" &&
      a.timestamp < issue.in_progress_timestamp
    )

source:
  document: Jira Development Standard
  version: "4.1"
  section: "3.2.1"
```

El proceso de consultar la fuente de datos, procesarla y normalizarla en forma de evidencia se documenta en el `evidence`:

```yaml
id: PR-001

subject:
  type: azure_devops.pull_request

evidence:
  - id: jira_key
    operation: extract_jira_reference
    from: subject

  - id: jira_issue
    operation: jira.get_issue
    args:
      key: ${jira_key}

  - id: jira_comments
    operation: jira.get_comments
    args:
      key: ${jira_key}

requirement:
  evaluator: cel
  expression: >
    jira_issue.status == "Review" &&
    jira_comments.exists(
      c,
      c.body.contains(subject.url)
    )
```

Los ejemplos de RASE son a modo de inspiración y no deben tomarse al pie de la letra ni sintáctica, ni semánticamente. El diseño de la solución final dará la sintaxis final.
Puede ser posible combinar fuentes:

```yaml
evidence_graph:

  nodes:

    pr:
      source: azure_devops
      entity: pull_request
      input: subject

    jira:
      source: jira
      entity: issue

    comments:
      source: jira
      entity: comments

  edges:

    - from: pr
      to: jira
      resolver:
        type: reference
        extractor: jira_issue_key
        fields:
          - title
          - description
          - source_branch

    - from: jira
      to: comments
      resolver:
        type: relation
        operation: jira.get_comments
```

Considera también el poder reaprovechar el Evidence Collection para varios controles a la vez.
No es decisión tomada pero me gusta Substrait para la parte del Evidence Collection (la parte de transformación de la fuente y normalización de la evidencia).

La Evidence Collection también se puede definir en el control:

```yaml
id: PR-001

subject:
  relation: github.pull_requests

evidence:
  query:
    from: github.pull_requests

    lateral_join:
      - relation: jira.issues
        on:
          jira.key: extract_jira_key(github.pr)

      - relation: jira.comments
        on:
          jira.comments.issue_key: jira.key

    project:
      - github.pr.url
      - jira.key
      - jira.status
      - jira.comments.body

requirement:
  evaluator: cel
  expression: >
    evidence.jira_key != null &&
    evidence.jira_status == "Review" &&
    evidence.comments.exists(
        c,
        normalize_url(c.body).contains(
            normalize_url(evidence.pr_url)
        )
    )
```

Aunque yo dé instrucciones en castellano traduce todo al inglés
El interfaz de usuario (CLI) es el cliente de comandos actual de Vatuta. Pero prepara el desarrollo para que pueda ser invocable como servicio MCP. Si hay que usar nuevas tecnologías o librerías que no sean Python no dudes en usarlas

La interpretabilidad de todo el proceso es muy importante. Para cada ejecución, en el directorio de log, en un directorio por invocación de tal manera que su nombre permite identificar la invocación se deja toda la información que permita seguir y auditar el razonamiento y todos los datos intermedios. Dentro de ese directorio cada stage del LLM flow debe ser otro directorio adicional. Dentro de cada directorio se deben almacenar de manera human readable, pero reutilizable por máquina: prompts y su respuesta, invocaciones a terceros sistemas y su respuesta...
