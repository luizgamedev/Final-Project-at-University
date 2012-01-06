/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package TestBidimensional;

import jade.core.Agent;
import jade.core.Profile;
import jade.core.ProfileImpl;
import jade.core.Runtime;
import jade.wrapper.AgentController;
import jade.wrapper.ContainerController;

/**
 *
 * @author kazenotenshi
 */
public class MainExecution{
    public static final int NAgents = 50;
    //public static long tempos[] = new long[NAgents];

    //public static AgentController ac = null;
    //public static ContainerController cc = null;
    //public static Runtime rt = null;

    
    public static void main(String args[]){
        

        String host="127.0.0.1",port="1099",name="Agent",mainClass = "TestBidimensional.Person";
        Runtime rt = Runtime.instance();
        Profile p = new ProfileImpl();
        p.setParameter(Profile.MAIN_HOST, host);
        p.setParameter(Profile.MAIN_PORT, port);


        //System.out.println(p.isLocalHost(host));
        //System.out.println("Initializing the Whole Thing!");
        ContainerController cc = rt.createMainContainer(p);
        
        if(cc != null){
            
            try{
                
                
                for(int i=0; i<NAgents ; i++){
                    
                    AgentController ac = cc.createNewAgent(i+"", mainClass, null);
                    ac.start();
                }
                


            }
            catch(Exception e){
                e.printStackTrace();
            }
        }



        //return cc;
    }
    

    
   
}
