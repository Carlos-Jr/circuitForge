module LogicModule ( 
    pi0, pi1, pi2, pi3, pi4, pi5, pi6, pi7,
    po0 );
  input  pi0, pi1, pi2, pi3, pi4, pi5, pi6, pi7;
  output po0;
  wire n0,n1,n2,n3,n4,n5;
  assign n0 = pi0 & pi1;
  assign n1 = pi2 & n0;
  assign n2 = pi3 & n1;
  assign n3 = pi4 & n2;
  assign n4 = pi5 & n3;
  assign n5 = pi6 & n4;
  assign po0 = pi7 & n5;

endmodule