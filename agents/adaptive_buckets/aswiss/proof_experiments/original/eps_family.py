import numpy as np
from search_pathology import maximin, post_split_P1, post_split_P2, expl_P1, expl_P2, W
A1 = np.array([[3,-3],[-3,2]], float); D = np.array([[-1,0],[0,0]], float)
for eps in (1.0, 0.3, 0.1, 0.03, 0.01, 0.001):
    A2 = A1 + eps*D; As = (A1, A2)
    vG = sum(W[k]*maximin(As[k])[0] for k in range(2))
    Abar = W[0]*A1 + W[1]*A2; _, x = maximin(Abar); _, y = maximin(-Abar.T)
    pre = expl_P1((x,x),As,vG) + expl_P2((y,y),As,vG)
    g = [W[k]*(As[k] @ y - x @ As[k] @ y) for k in range(2)]
    labels = [int(np.argmax(gk)) for gk in g]
    _, x1, x2 = post_split_P1(A1, A2, "best"); yp = post_split_P2(A1, A2)
    post = expl_P1((x1,x2),As,vG) + expl_P2((yp,yp),As,vG)
    print(f"eps={eps:<6} B0(pre)={pre:.5f}  B1(post, best case)={post:.4f}  ratio={post/pre:9.1f}  "
          f"labels={labels} max adv={[round(gk.max(),5) for gk in g]}")
