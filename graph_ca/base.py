"""
Итак ... что же мы здесь будем делать ...

Здесь мы будем исследовать перспективы развития графовых клеточных автоматов. Как известно, граф это просто набор
"вершин" и "ребер". Он может быть направленым, ненаправленным, цикличным, или не цикличным и т.д. Один из способов
описания графов является так называемая "матрица смежности". Это квадратная матрица формата shape=[N, N],
где N - это количество вершин в графе, заполненная нулями и еденицами. При рассмотрении некой конкретный "i-ой" вершины,
мы рассматриваем столбец этой матрицы с индексом "i", этот столбец формат [1, N], вершина "i" связаня с теми вершинами,
для которых значения текущего столбца не нулевое.

Значения матрицы смежности могут иметь знак, он говорит о направлении связи, если все элементы матрицы смежности
одного знака, то не имеет смысла говорить о направлении связей. Значения могут быть как целыми, так и дробными,
как равными еденице, так о отличаться от нее. В таком случае считается что у данного ребра есть удельный вес, равный
значению ненулевого элемента в матрице смежности. Мне не известно существуют ли графы, при описании которых изпользуются
комплексные числа, поэтому будем считать, что все значения матрицы смежности действительные. Если у матрицы ненулевые
диагональные элементы, то значит что в графе существуют верщины которые имеют связь сами с собой.

И вот теперь мы хотим сделать клеточный автомат, который будет представлять из себя по сути дела обычный граф.
Его состояния будут описывать как матрицей смежности, описывающей просто топологию графа, так еще и надобором
дополнительных атрибутов. Например вершины графа могут быть в неком состоянии, и для каждой вершины будет задан
диапазон, дискретный или непрерывный, возможных состояний. Рёбрам можно добавить направление, и т.д.
Здесь мы ограничены только нашей фантазией и вычислительными ресурсами. После того как будет опередено "структура" /
"диапазон состояний" клеточного автомата, потребуется определить диапазон его возможных действий, или правила
обновления клеточного автомата. По сути своей это функция отображения текущего сотояния конкретной "клетки",
в значение из определенного диапазона значений. Эта функция может быть табличной, дискретной, или непрерывной.
Она может быть как дифференцируемой, так и недефференцируемой. И данная функция должна быть для всех клеток одинакова.

И после того как будет созданы классы различных клеточных автоматов и их действий, можно будет запускать моделирование
развития клеточного автомата, и смотреть что получится в итоге.

ВАЖНОЕ ЗАМЕЧАНИЕ: Правила обновления могут касаться не только вершин графа, но и отдельных подграфов. Это было бы очень
важным условием роста. Идея взята из работы Стефана Вольврама:
https://writings.stephenwolfram.com/2020/04/finally-we-may-have-a-path-to-the-fundamental-theory-of-physics-and-its-beautiful/

Дедок может и сошел с ума, но на его графиках можно видеть как из очень простого графа вырастают очень сложные
структуры, не за счет того что, от конкретной вершины рождается новая, а от того, что он согласно определенному правилу,
меняет один подграф на другой. И такой вариант развития событий крайне инетересно было бы потыкать. К физике это не
применимо, но может подобным образом удасться наладить поиск архитектуры нейронной сети, или вести поиск локального
правила обновления весов. И таким образом поискать замену халеному алгоритму обратного распространения ошибки.

На самом деле смысл всей этой затеи именно в этом. Я хочу создать более крутую альтернативу существующему методу
обучения нейронных сетей, и желательно сделать его независимым от размеченных данных. В этом смысле интересно посмотреть
на данную проблему именно с этой точки зрения.

Ну ... Сказать что планы грандиозные - ничего не сказать. Но это не значит что я не могу попытаться.
Ибо путь в тысячу ли, начинается с первого шага. А оформить кодовую базу под эти эксперименты мне уж точно под силу.
"""