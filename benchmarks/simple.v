//Circuito de exemplo de Designing Partially Reversible FCN Circuits (Chaves,2019)
module LogicModule (
    pi0, pi1
    po0, po1
);
    input  pi0, pi1;
    output po0, po1;
    wire   n1,n2,n3;

    assign n1 = pi0 & pi1;
    assign n2 = pi0 & ~n1;
    assign n3 = pi1 & ~n1;
    assign po0 = ~n2 & ~n3;
    assign po1 = ~n3 & pi1;
endmodule
