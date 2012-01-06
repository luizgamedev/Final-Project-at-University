/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package releaseAMS;

import jade.core.Profile;
import jade.core.ProfileImpl;
import jade.core.Runtime;
import jade.wrapper.AgentController;
import jade.wrapper.ContainerController;

/**
 *
 * @author kazenotenshi
 */
public class MainAgents{
    public static final int NAgents = 5000;
    //public static long tempos[] = new long[NAgents];
    public static void main(String args[]){
        String host=/*"macbook-pro-de-luiz.local"*/"localhost",port="1099",name="Agent",mainClass = "releaseAMS.Person";

        Runtime rt = Runtime.instance();
        Profile p = new ProfileImpl();

        p.setParameter(Profile.MAIN_HOST, host);
        p.setParameter(Profile.MAIN_PORT, port);

        
        System.out.println(p.isLocalHost(host));

        ContainerController cc = rt.createMainContainer(p);
        //ContainerController ca = rt.createAgentContainer();


        if(cc != null){
            
            try{
                
                for(int i=0; i<NAgents ; i++){
                    AgentController ac = cc.createNewAgent(i+"", mainClass, null);

                    ac.start();

                    //Thread.sleep(1000);
                }
                
                
            }
            catch(Exception e){
                e.printStackTrace();
            }
        }
      //  long maior=0;
      //  for(int i=0; i<NAgents ; i++){
      //      if(tempos[i]>maior){
      //          maior = tempos[i];
      //      }
      //  }
      //  System.out.println("Maior Tempo = "+maior);

    }
}
