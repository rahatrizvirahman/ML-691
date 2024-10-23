import math

def dotting(f):
    var = []
    dotted = []

    x = f[0][1]

    for i, operation in enumerate(f):
        op = operation[0]
       
        if op == 'X':
            var.append(operation[1])
            dotted.append(1.0)
        elif op == 'C':
            var.append(operation[1])
            dotted.append(0.0)
        elif op == '+':
            u, v = operation[1], operation[2]
            var.append(var[u] + var[v])
            dotted.append(dotted[u] + dotted[v])
        elif op == '-': 
            u, v = operation[1], operation[2]
            var.append(var[u] - var[v])
            dotted.append(dotted[u] - dotted[v])
        elif op == '*':
            u, v = operation[1], operation[2]
            var.append(var[u] * var[v])
            dotted.append(dotted[u] * var[v] + var[u] * dotted[v])
        elif op == '/':
            u, v = operation[1], operation[2]
            var.append(var[u] / var[v])
            dotted.append((dotted[u] * var[v] - var[u] * dotted[v]) / (var[v] ** 2))
        elif op == 'S': 
            u = operation[1]
            var.append(var[u] ** 2)
            dotted.append(2.0 * var[u] * dotted[u])
        elif op == 'E': 
            u = operation[1]
            var.append(math.exp(var[u]))
            dotted.append(dotted[u] * math.exp(var[u]))
        elif op == 'L':  
            u = operation[1]
            var.append(math.log(var[u]))
            dotted.append(dotted[u] / var[u])
       

    y = var[-1]
    dotted_y = dotted[-1]

    return x, y, dotted_y

def repeated_dotting(f, step_count=10):
    x = f[0][1]

    for i in range(step_count):
        updated_f = []
        
        if i>0:
            for j in range(len(f)):
                if j==0:
                    updated_f.append(('X', x))
                else:
                    updated_f.append(f[j])
        else:
            updated_f = f

        x, y, dotted_y = dotting(updated_f)
        x -= 0.001 * dotted_y  

    return y, dotted_y

if __name__ == "__main__":
    f = [('X', 2.0),
         ('C', 1.0),
         ('+', 0, 1),
         ('*', 0, 2),
         ('S', 3)]
    
    print(dotting(f))
    print(repeated_dotting(f))
