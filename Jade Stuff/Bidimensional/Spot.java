/*
 * Universidade Federal Fluminense
 * Pólo Universitário de Rio das Ostras
 * Faculdade Federal de Rio das Ostras
 * Departamento de Ciência e Tecnologia
 * Projeto Final II
 *
 * @author Luiz Guilherme Oliveira dos Santos
 */

package TestBidimensional;


public class Spot extends WorldMap{

    public boolean isInicialSpot;

    public Spot(boolean type){
        isInicialSpot = type;
    }

    public Spot(){

    }

    @Override
    public String toString(){
        return "0";
    }

}
