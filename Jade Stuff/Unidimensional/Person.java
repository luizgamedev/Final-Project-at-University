/*
 * Universidade Federal Fluminense
 * Faculdade Federal de Rio das Ostras
 * Departamento de Ciência e Tecnologia
 * Ciência da Computação
 *
 * Projeto Final
 *
 */

package releaseAMS;

import jade.content.ContentManager;
import jade.core.Agent;
import java.util.logging.Level;
import java.util.logging.Logger;
import jade.content.onto.basic.Action;
import jade.core.Agent;
import jade.core.ContainerID;
import jade.core.behaviours.Behaviour;
import jade.domain.FIPANames;
import jade.domain.JADEAgentManagement.CreateAgent;
import jade.domain.JADEAgentManagement.JADEManagementOntology;
import jade.domain.introspection.ACLMessage;
import jade.proto.AchieveREInitiator;
import java.util.Calendar;
import java.util.Random;
/**
 *
 * @author Luiz Guilherme Oliveira dos Santos
 */

public class Person extends Agent{
    private static final int maxPackages = 10000;

    public static Packages pk = new Packages(maxPackages);

    long init;
    long end;
    long diff;

    public Person(){
        init = System.currentTimeMillis();
    }


    protected void setup(){

        //imprimindo o nome do agente
        //System.out.println("Agent "+getLocalName()+" started.");

        //Adicionando o comportamento generico
        addBehaviour(new PersonBehaviour(pk));
    

    }



    private class PersonBehaviour extends Behaviour{
        public Packages pk;
        public int pos;
        private Random rd = new Random();

        public PersonBehaviour(Packages pk){
            this.pk = pk;
            pos = rd.nextInt() % pk.numberOfPackages;
                if(pos < 0) pos *=-1;

        }


        public void action(){

            if(pk.seePosition(pos) == true){
                pk.getPackage(getLocalName(), pos);
            }

            else{
                int dir = ((pos+1) % (pk.numberOfPackages));

                int esq = ((pos-1) % (pk.numberOfPackages));
                    if(esq == -1)
                        esq = pk.numberOfPackages - 1;

                //Olhar a direita
                if(pk.seePosition(dir) == true){
                    pos = dir;
                    //System.out.println(getLocalName() + ": Andei para a Direita");
                }
                //olhar a esquerda
                else
                    if(pk.seePosition(esq) == true){
                        pos = esq;
                       // System.out.println(getLocalName() + ": Andei para a Esquerda");
                    }
                else{

                        int t = rd.nextInt() % 2;
                        if(t == 1){
                            pos = dir;
                            //System.out.println(getLocalName() + ": Andei para a Direita");
                        }
                        else{
                            pos = esq;
                            //System.out.println(getLocalName() + ": Andei para a Esquerda");
                        }

                }
                //Tempo de espera de andar
                //try {
                //    Thread.sleep(pk.quantum);
                //} catch (InterruptedException ex) {
                //    Logger.getLogger(Person.class.getName()).log(Level.SEVERE, null, ex);
                //}
            }

        }


        public boolean done() {
            //System.out.println(getLocalName()+": Is it done? "+pk.allPackagesAreGone());
            boolean test = pk.allPackagesAreGone();
            if(test == true){
                end = System.currentTimeMillis();
                diff = end-init;

                //MainAgents.tempos[Integer.parseInt(getLocalName())] = diff;
                System.out.println("Diferenca do " + getLocalName() + " = "+ diff );
            }
            return test;
        }

        public void doDelete(){
            
        }

    }
}
