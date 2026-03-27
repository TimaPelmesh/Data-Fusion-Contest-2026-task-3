# Полная система уравнений MIP-модели для первого дня

## Множества и индексы

Герои:

$$
K = \{1,\dots,m\}
$$

где $m$ — число фиксированных героев

Объекты первого дня:

$$
V = \{1,\dots,n\}
$$

где $n$ — число объектов дня 1

Дуги между объектами:

$$
A = \{(i,j)\in V\times V \mid i \ne j\}
$$

## Параметры

Ёмкость героя $k$:

$$
c_k
$$

Расстояние от таверны до объекта $i$:

$$
s_i
$$

Расстояние между объектами $i$ и $j$:

$$
d_{ij}
$$

Стоимость посещения:

$$
V_c = 100
$$

Допустимые стартовые объекты для героя $k$:

$$
S_k = \{i \in V \mid s_i + 1 \le c_k\}
$$

Маленький коэффициент tie-breaker:

$$
\varepsilon > 0
$$

## Переменные

### Бинарные

Активность героя:

$$
a_k \in \{0,1\} \qquad \forall k \in K
$$

Посещение объекта:

$$
v_{i,k} \in \{0,1\} \qquad \forall i \in V, k \in K
$$

Выбор дуги:

$$
x_{i,j,k} \in \{0,1\} \qquad \forall (i,j)\in A, k \in K
$$

### Непрерывные

MTZ-переменные:

$$
u_{i,k} \ge 0 \qquad \forall i \in V, k \in K
$$

## Вспомогательные выражения

Входящая степень вершины:

$$
in_{i,k} = \sum_{p:(p,i)\in A} x_{p,i,k}
$$

Исходящая степень вершины:

$$
out_{i,k} = \sum_{s:(i,s)\in A} x_{i,s,k}
$$

Число посещённых объектов героем:

$$
visit_k = \sum_{i\in V} v_{i,k}
$$

Число стартов:

$$
start_k = \sum_{i\in V}(v_{i,k} - in_{i,k})
$$

Число концов:

$$
end_k = \sum_{i\in V}(v_{i,k} - out_{i,k})
$$

Стартовая стоимость:

$$
startCost_k = \sum_{i\in V} s_i\,(v_{i,k} - in_{i,k})
$$

Стоимость дуг:

$$
arcCost_k = \sum_{(i,j)\in A} d_{ij}\,x_{i,j,k}
$$

Полная стоимость маршрута героя:

$$
used_k = startCost_k + arcCost_k + V_c \cdot visit_k - (V_c - 1)a_k
$$

Leftover героя:

$$
leftover_k = c_k a_k - used_k
$$

## Целевая функция

$$
\max \left(
\sum_{k\in K}\sum_{i\in V} v_{i,k}
+
\varepsilon \sum_{k\in K} leftover_k
\right)
$$

То есть:
1. сначала хотим посетить как можно больше объектов;
2. среди равных решений предпочитаем больше leftover.

## Ограничения

### 1. Каждый объект можно посетить не более одного раза

$$
\sum_{k\in K} v_{i,k} \le 1
\qquad \forall i\in V
$$

### 2. Если герой не активен, он не может посещать объекты

$$
\sum_{i\in V} v_{i,k} \le n\,a_k
\qquad \forall k\in K
$$

### 3. У активного героя ровно один старт

$$
\sum_{i\in V}(v_{i,k} - in_{i,k}) = a_k
\qquad \forall k\in K
$$

### 4. У активного героя ровно один конец

$$
\sum_{i\in V}(v_{i,k} - out_{i,k}) = a_k
\qquad \forall k\in K
$$

### 5. Вход через объект возможен только если объект посещён

$$
in_{i,k} \le v_{i,k}
\qquad \forall i\in V, k\in K
$$

### 6. Выход через объект возможен только если объект посещён

$$
out_{i,k} \le v_{i,k}
\qquad \forall i\in V, k\in K
$$

### 7. Запрет недопустимого старта

Если объект не может быть стартом героя $k$, то он должен иметь входящую дугу:

$$
v_{i,k} \le in_{i,k}
\qquad \forall k\in K, \forall i\in V \setminus S_k
$$

### 8. Ограничение по очкам хода

$$
used_k \le c_k
\qquad \forall k\in K
$$

то есть

$$
\sum_{i\in V} s_i(v_{i,k}-in_{i,k})
+
\sum_{(i,j)\in A} d_{ij}x_{i,j,k}
+
V_c \sum_{i\in V} v_{i,k}-(V_c-1)a_k
\le c_k
\qquad \forall k\in K
$$

### 9. Нижняя граница на MTZ-переменные

$$
u_{i,k} \ge v_{i,k}
\qquad \forall i\in V, k\in K
$$

### 10. Верхняя граница на MTZ-переменные

$$
u_{i,k} \le n\,v_{i,k}
\qquad \forall i\in V, k\in K
$$

### 11. MTZ-ограничения против подциклов

$$
u_{i,k} - u_{j,k} + n\,x_{i,j,k} \le n-1
\qquad \forall (i,j)\in A, k\in K
$$

## Области значений переменных

$$
a_k \in \{0,1\}
\qquad \forall k\in K
$$

$$
v_{i,k} \in \{0,1\}
\qquad \forall i\in V, k\in K
$$

$$
x_{i,j,k} \in \{0,1\}
\qquad \forall (i,j)\in A, k\in K
$$

$$
u_{i,k} \ge 0
\qquad \forall i\in V, k\in K
$$
