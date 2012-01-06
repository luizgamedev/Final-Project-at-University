/*
 * Universidade Federal Fluminense
 * Faculdade Federal de Rio das Ostras
 * Departamento de Ciência e Tecnologia
 * Ciência da Computação
 *
 * Projeto Final
 *
 */

package TestBidimensional;


import jade.core.Agent;
import java.util.logging.Level;
import java.util.logging.Logger;

import jade.core.Agent;

import jade.core.behaviours.Behaviour;
import java.util.ArrayList;
import java.util.Iterator;

 /*
 * @author Luiz Guilherme Oliveira dos Santos
 */

public class Person extends Agent{

    //************************************************************************
    //All the Constants and Global Variables
    //************************************************************************
    
    private static int sceneNumber = 0;
    
    public static Scenario sc = new Scenario(sceneNumber);
    
    
    private Point rootPos;

    private Point checkPointPos;

    private WorldMap[][] myMap;

    private final Point NORTH = new Point(0,1);
    private final Point SOUTH = new Point(0,-1);
    private final Point EAST = new Point(1,0);
    private final Point WEST = new Point(-1,0);
    private final Point NORTHEAST = new Point(1,1);
    private final Point NORTHWEST = new Point(-1,1);
    private final Point SOUTHEAST = new Point(1,-1);
    private final Point SOUTHWEST = new Point(-1,-1);

    

    private long init;
    private long end;
    private long diff;


    
    
    
    //************************************************************************
    //Constructor of The Class
    //************************************************************************

    public Person(){
 
        
    }

    //************************************************************************
    //Method Setup, This method initializes all variables and behaviours
    //************************************************************************

    @Override
    protected void setup(){
        init = System.currentTimeMillis();

        myMap = sc.getMap();

        rootPos = sc.getMyPosition(getLocalName());

        checkPointPos = sc.getCheckPoint();

        //System.out.println("Inside the Agent " + getLocalName() + "!\n");
        //if(sc != null)
        //    System.out.println("Agent " + getLocalName() + " recognizes the scenario! \n");
        //else
        if(sc == null)
            return;
        

        //System.out.println("Agent " + getLocalName() + " wants to know my position and my checkpoint \n");
        

        
        addBehaviour(new PersonBehaviour(rootPos, checkPointPos, sc, myMap));




    }


    //************************************************************************
    //Private Class Person Behaviour, this objects rules the agents behaviour
    //************************************************************************

    private class PersonBehaviour extends Behaviour{

        //************************************************************************
        //Global Variables and Constants of PersonBehaviour
        //************************************************************************

        public Point rootPos = null;
        public Point checkPointPos;
        public Scenario sc;
        public WorldMap[][] myMap;

        private ArrayList<Point> myRoute;
        private int myRouteIndex = 0;
        

        //************************************************************************
        //Constructor of The Class
        //************************************************************************
        
        public PersonBehaviour(Point rootPos, Point checkPointPos, Scenario sc, WorldMap[][] myMap){

            this.rootPos = rootPos;
            this.checkPointPos = checkPointPos;
            this.myMap = myMap;
            this.sc = sc;

            //System.out.println("Agent "+getLocalName()+" Starting Motherfucka!\n");

            
            //System.out.println("this.rootPos is null? "+this.rootPos);
            if(this.rootPos != null){
                //System.out.println("Agent "+getLocalName()+" Calculating the Route\n");
                traceRoute();
            }

            
            
        }


        

        //************************************************************************
        //Trace Route Method, Calculates de A* Heuristic
        //************************************************************************
        
        public void traceRoute(){
            
            ArrayList<AStarNode> openList = new ArrayList();
            ArrayList<AStarNode> closedList = new ArrayList();

            //Acrescentando pontos adjacentes do Ponto inicial na lista aberta e colocando o ponto inicial na lista fechada
            AStarNode root = new AStarNode(rootPos);
            root.father = null;
            root.F = 0;
            root.G = 0;
            root.H = 0;
            

            openList = addAdjacents(root, openList);
            closedList.add(root);


            while(openList.size() > 0){
                
                AStarNode A = openList.get(0);
                //System.out.println("Agent "+getLocalName()+" Gettin' Point " + A.thisPoint);
                //System.out.println("Now Getting point "+A.thisPoint);
                //openList.remove(0);
                
                closedList.add(A);

                //Condição de Saída
                if(isInsideTheList(closedList, checkPointPos)){
                    
                    break;
                }


                //A = openList.get(0);
                openList.remove(0);

                openList = addAdjacents(A, openList);
                //System.out.println("HERE");
                
                
            }

            
            //System.out.println("HERE");
            //To do the Finish:

            ArrayList<AStarNode> listTemp = new ArrayList<AStarNode>();
            AStarNode k = closedList.get(closedList.size() - 1);
            closedList.remove(closedList.size() - 1);
            while(!(k.thisPoint.equals(rootPos))){
                
                listTemp.add(k);
                k = k.father;

            }

            myRoute = new ArrayList<Point>();
            for(int i = listTemp.size() - 1 ; i > -1 ; i--){
                myRoute.add(listTemp.get(i).thisPoint);
                listTemp.remove(i);
            }

            
            //System.out.println("Agent "+getLocalName()+" add on the ready list!");
            sc.readyAgents.add(getLocalName());

        }

        //************************************************************************
        //The next methods are auxiliar methods of TraceRoute
        //************************************************************************

        private boolean isInsideTheList(ArrayList<AStarNode> l, Point p){
            Iterator<AStarNode> it = l.iterator();
            while(it.hasNext()){
                AStarNode temp = it.next();
                if(temp.thisPoint.equals(p)){
                    return true;
                }
            }
            return false;
        }

        

        private ArrayList<AStarNode> addAdjacents(AStarNode p, ArrayList<AStarNode> actualList){

            Point temp = new Point();
            temp.x = p.thisPoint.x;
            temp.y = p.thisPoint.y;
            //System.out.println("Temp = "+temp);
            
            //System.out.println(rootPos);
            //North
            temp.soma(NORTH);
            //System.out.println("AGENT "+getLocalName()+" Temp+NORTH = "+temp);
            if(temp.x > -1 && temp.x < myMap.length){
                if(temp.y > -1 && temp.y < myMap[temp.x].length){
                    if(!(myMap[temp.x][temp.y].toString().equals("1"))){
                        AStarNode A = new AStarNode(temp);
                        A.father = p;
                        A.G = 10;
                        A.H = calculateH(temp,checkPointPos);
                        A.F = A.G + A.H;
                        //if(A.thisPoint.x == 0 && A.thisPoint.y == 0) System.out.println("AGENT "+getLocalName()+" ACHEI O 0,0!!!");
                        if(!inTheList(A,actualList)) actualList.add(A);

                    }
                }
            }


            //South
            temp = new Point();
            temp.x = p.thisPoint.x;
            temp.y = p.thisPoint.y;
            temp.soma(SOUTH);
            //System.out.println("AGENT "+getLocalName()+" Temp+SOUTH = "+temp);
            if(temp.x > -1 && temp.x < myMap.length){
                if(temp.y > -1 && temp.y < myMap[temp.x].length){
                    if(!(myMap[temp.x][temp.y].toString().equals("1"))){
                        AStarNode A = new AStarNode(temp);
                        A.father = p;
                        A.G = 10;
                        A.H = calculateH(temp,checkPointPos);
                        A.F = A.G + A.H;
                        //if(A.thisPoint.x == 0 && A.thisPoint.y == 0) System.out.println("AGENT "+getLocalName()+" ACHEI O 0,0!!!");
                        if(!inTheList(A,actualList)) actualList.add(A);

                    }
                }
            }

            //East
            temp = new Point();
            temp.x = p.thisPoint.x;
            temp.y = p.thisPoint.y;
            temp.soma(EAST);
           // System.out.println("AGENT "+getLocalName()+" Temp+EAST = "+temp);
            if(temp.x > -1 && temp.x < myMap.length){
                if(temp.y > -1 && temp.y < myMap[temp.x].length){
                    if(!(myMap[temp.x][temp.y].toString().equals("1"))){
                        AStarNode A = new AStarNode(temp);
                        A.father = p;
                        A.G = 10;
                        A.H = calculateH(temp,checkPointPos);
                        A.F = A.G + A.H;
                        //if(A.thisPoint.x == 0 && A.thisPoint.y == 0) System.out.println("AGENT "+getLocalName()+" ACHEI O 0,0!!!");
                        if(!inTheList(A,actualList)) actualList.add(A);

                    }
                }
            }

            //West
            temp = new Point();
            temp.x = p.thisPoint.x;
            temp.y = p.thisPoint.y;
            temp.soma(WEST);
           // System.out.println("AGENT "+getLocalName()+" WEST = "+temp);
            if(temp.x > -1 && temp.x < myMap.length){
                if(temp.y > -1 && temp.y < myMap[temp.x].length){
                    if(!(myMap[temp.x][temp.y].toString().equals("1"))){
                        AStarNode A = new AStarNode(temp);
                        A.father = p;
                        A.G = 10;
                        A.H = calculateH(temp,checkPointPos);
                        A.F = A.G + A.H;
                        //if(A.thisPoint.x == 0 && A.thisPoint.y == 0) System.out.println("AGENT "+getLocalName()+" ACHEI O 0,0!!!");
                        if(!inTheList(A,actualList)) actualList.add(A);

                    }
                }
            }

            //Northwest
            temp = new Point();
            temp.x = p.thisPoint.x;
            temp.y = p.thisPoint.y;
            temp.soma(NORTHWEST);
            //System.out.println("AGENT "+getLocalName()+" Temp+NORTHWEST = "+temp);
            if(temp.x > -1 && temp.x < myMap.length){
                if(temp.y > -1 && temp.y < myMap[temp.x].length){
                    if(!(myMap[temp.x][temp.y].toString().equals("1"))){
                        AStarNode A = new AStarNode(temp);
                        A.father = p;
                        A.G = 14;
                        A.H = calculateH(temp,checkPointPos);
                        A.F = A.G + A.H;
                        //if(A.thisPoint.x == 0 && A.thisPoint.y == 0) System.out.println("AGENT "+getLocalName()+" ACHEI O 0,0!!!");
                        if(!inTheList(A,actualList)) actualList.add(A);

                    }
                }
            }

            //Southwest
            temp = new Point();
            temp.x = p.thisPoint.x;
            temp.y = p.thisPoint.y;
            temp.soma(SOUTHWEST);
            //System.out.println("AGENT "+getLocalName()+" Temp+SOUTHWEST = "+temp);
            if(temp.x > -1 && temp.x < myMap.length){
                if(temp.y > -1 && temp.y < myMap[temp.x].length){
                    if(!(myMap[temp.x][temp.y].toString().equals("1"))){
                        AStarNode A = new AStarNode(temp);
                        A.father = p;
                        A.G = 14;
                        A.H = calculateH(temp,checkPointPos);
                        A.F = A.G + A.H;
                        //if(A.thisPoint.x == 0 && A.thisPoint.y == 0) System.out.println("AGENT "+getLocalName()+" ACHEI O 0,0!!!");
                        if(!inTheList(A,actualList)) actualList.add(A);

                    }
                }
            }

            //Southeast
            temp = new Point();
            temp.x = p.thisPoint.x;
            temp.y = p.thisPoint.y;
            temp.soma(SOUTHEAST);
            //System.out.println("AGENT "+getLocalName()+" Temp+SOUTHEAST = "+temp);
            if(temp.x > -1 && temp.x < myMap.length){
                if(temp.y > -1 && temp.y < myMap[temp.x].length){
                    if(!(myMap[temp.x][temp.y].toString().equals("1"))){
                        AStarNode A = new AStarNode(temp);
                        A.father = p;
                        A.G = 14;
                        A.H = calculateH(temp,checkPointPos);
                        A.F = A.G + A.H;
                        //if(A.thisPoint.x == 0 && A.thisPoint.y == 0) System.out.println("AGENT "+getLocalName()+" ACHEI O 0,0!!!");
                        if(!inTheList(A,actualList)) actualList.add(A);

                    }
                }
            }

            //northeast
            temp = new Point();
            temp.x = p.thisPoint.x;
            temp.y = p.thisPoint.y;
            temp.soma(NORTHEAST);
            //System.out.println("AGENT "+getLocalName()+" Temp+NORTHEAST = "+temp);
            if(temp.x > -1 && temp.x < myMap.length){
                if(temp.y > -1 && temp.y < myMap[temp.x].length){
                    if(!(myMap[temp.x][temp.y].toString().equals("1"))){
                        AStarNode A = new AStarNode(temp);
                        A.father = p;
                        A.G = 14;
                        A.H = calculateH(temp,checkPointPos);
                        A.F = A.G + A.H;
                        //if(A.thisPoint.x == 0 && A.thisPoint.y == 0) System.out.println("AGENT "+getLocalName()+" ACHEI O 0,0!!!");
                        if(!inTheList(A,actualList)) actualList.add(A);

                    }
                }
            }

            //Falta Ordenar a Lista
            qSort(actualList, 0, actualList.size()-1);
            
            return actualList;
        }

        private int calculateH(Point P, Point To){
            
            int t1 = To.x - P.x;
            int t2 = To.y - P.y;

            if(t1 < 0) t1 *= -1;
            if(t2 < 0) t2 *= -1;
            //System.out.println("Agent "+getLocalName()+" Calculating H of "+P+" and "+To+" = "+(t1+t2));
            return t1+t2;
        }
        
        //************************************************************************
        //QuickSort of An ArrayList
        //************************************************************************
        public void qSort(ArrayList<AStarNode> l, int low, int high)
        {
            // region of array to be sorted [low, high]

            int i = low, j = high;
            AStarNode h;
            AStarNode x = l.get(low);
            //l.remove(low);
            //  partition
            do
            {
                // find left index
                while (l.get(i).F < x.F)
                {
                    //l.remove(i);
                    i++;
                }
                // find right index
                while (l.get(j).F > x.F)
                {
                    //l.remove(j);
                    j--;
                }
                // swapping numbers at specified indexes when necessary
                if (i<=j)
                {
                    h = l.get(i);
                    l.set(i, l.get(j));
                    l.set(j, h);
                    i++;
                    j--;
                }
            } while (i<=j);

            //  recursion
            if (low<j)
                qSort(l, low, j);
            if (i<high)
                qSort(l, i, high);
        }// quicksort method

        
        private boolean inTheList(AStarNode p, ArrayList<AStarNode> list){


            for(int i = 0; i<list.size() ; i++){
                if(list.get(i).thisPoint.equals(p.thisPoint)){

                    AStarNode temp = list.get(i);
                    if(temp.G > p.G){
                        list.set(i,p);
                        
                    }

                    return true;
                }
            }



            return false;
        }

        //A* Subfuncitions Ends /o/


        //************************************************************************
        //Action Method, Says What are the actions of an Agent
        //************************************************************************
        public void action(){
            //action of the agent
            //System.out.println("HERE!");
            boolean tryToMove;
            if(rootPos != null)
                tryToMove = sc.setMyNewPos(getLocalName(), rootPos, myRoute.get(myRouteIndex));
            else{
                //System.out.println("Agent "+getLocalName()+" Not Allowed to enter in the Scenario");
                tryToMove = false;
            }
            
            if(tryToMove){
                //System.out.println("Agent "+getLocalName()+" moving to "+myRoute.get(myRouteIndex));
                rootPos = myRoute.get(myRouteIndex);
                myRouteIndex++;
            }
            else{
                //System.out.println("Agent "+getLocalName()+" standing still!");
            }
            
            

        }

        //************************************************************************
        //Done Method, Says if the agent is ready to finish
        //************************************************************************
        public boolean done() {
           
            if(rootPos == null){
                //System.out.println("Agent "+getLocalName()+": Sorry! I didn't enter ¬¬");
                return true;
                
            }

            if(rootPos.equals(checkPointPos) || myMap == null || myRouteIndex > (myRoute.size()-1)){
                //System.out.println("Agent "+getLocalName()+": I'm Done");
                doEnd();
                end = System.currentTimeMillis();
                diff = end-init;

                //MainAgents.tempos[Integer.parseInt(getLocalName())] = diff;
                System.out.println("Diferenca do Agente " + getLocalName() + " = "+ diff );
                return true;
            }
            
            else return false;


        }

        //************************************************************************
        //DoEnd, FISNISH HIM!
        //************************************************************************
        public void doEnd(){
            
            try {
                //System.out.println("Agent "+getLocalName()+" Finishing! /o/");
                if(rootPos != null) sc.killAgentPos(getLocalName(), myRoute.get((myRoute.size()-1)));
                this.finalize();
            } catch (Throwable ex) {
                Logger.getLogger(Person.class.getName()).log(Level.SEVERE, null, ex);
            }
            
        }

    }
}