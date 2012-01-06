/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package releaseAMS;

import jade.util.Logger;
import java.util.Random;
import java.util.logging.Level;

/**
 *
 * @author kazenotenshi
 */
public class Packages {
    private boolean packages[];
    protected final long quantum = 1750;
    public Random rd;
    protected int numberOfPackages;

    public Packages(int numberOfPackages){

        packages = new boolean[numberOfPackages];
        this.numberOfPackages = numberOfPackages;
        rd = new Random();

        for(int i=0 ; i<packages.length ; i++){
            packages[i] = rd.nextBoolean();
            //packages[i] = true;
        }


    }

    public void getPackage(String localName, int pos){
        //System.out.println(localName + ": Peguei o pacote da posição "+pos);
        packages[pos] = false;
        //try {
        //    Thread.sleep(quantum * 2);
        //} catch (InterruptedException ex) {
        //    Logger.getLogger(Person.class.getName()).log(Level.SEVERE, null, ex);
        //}
    }

    public boolean seePosition(int pos){
        return packages[pos];
    }

    public boolean allPackagesAreGone(){
        for(int i = 0 ; i < packages.length ; i++){
            if(packages[i] == true)
                return false;
        }
        return true;
    }

    public void printPackages(){
        System.out.print("{ ");
        for(int i = 0 ; i < packages.length ; i++){
            System.out.print(i+":"+packages[i]+" ");
        }
    }



}
