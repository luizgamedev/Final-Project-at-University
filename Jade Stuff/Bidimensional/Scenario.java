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



import java.util.Scanner;
import java.io.*;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.Random;


/**
 *
 * @author Luiz Guilherme Oliveira dos Santos
 * Projeto Final
 */

//Classe que implementa o cenário dos agentes
public class Scenario {
    private Scanner sc;
    private static WorldMap map[][];
    private int dim;
    
    private final String path = "/Volumes/KaZe_/UFF/2010 - 1/Projeto Final II/Bidiomensional Scenario/CUDA/";
    private final String types[] = {"cleanMapText", "corridorMapText", "outOfRoomMapText", "inTheRoomMapText"};
    private int MapType;
    private ArrayList<Point> avaliableSpots;
    private ArrayList<MapAgent> agents = null;
    private static Point checkPoint;
    private static int numAgents = 0;
    private static int maxAgents;
    private static boolean ready = false;

    
    public static ArrayList<String> readyAgents = new ArrayList<String>();
    

    public Scenario(int Maptype){
        this.MapType = Maptype;

        
        initializeMap("Test4");
        findAvaliableSpots();
        findCheckPoint();
        //System.out.println("Checkpoint = "+checkPoint);



    }

    private void findAvaliableSpots(){
        avaliableSpots = new ArrayList<Point>();
        for(int i = 0; i<dim ; i++){
            for(int j = 0; j<dim ; j++){
                if(map[i][j].getClass().toString().equals(Spot.class.toString())){
                    Spot s = (Spot)map[i][j];
                    if(s.isInicialSpot == true){
                        Point p = new Point(i,j);
                        avaliableSpots.add(p);
                    }
                }
            }
        }

        maxAgents = avaliableSpots.size();

        
        
    }

    private void findCheckPoint(){
        //avaliableSpots = new ArrayList<Point>();
        for(int i = 0; i<dim ; i++){
            for(int j = 0; j<dim ; j++){
                if(map[i][j].getClass().toString().equals(CheckPoint.class.toString())){
                    checkPoint = new Point(i,j);
                }
            }
        }

        

    }


    public synchronized Point getMyPosition(String name){
        
        if(agents == null){
            agents = new ArrayList<MapAgent>();
            
        }

        if(maxAgents > 0){
            
            Random r = new Random();


            int num = r.nextInt(maxAgents);

            

            MapAgent m = new MapAgent();
            m.name = name;
            m.pos = avaliableSpots.get(num);
            
            
            for(MapAgent p : agents){
                if(p.name.equals(name) == true)
                    return p.pos;
            }

            avaliableSpots.remove(num);
            agents.add(m);
            map[m.pos.x][m.pos.y] = m;

            maxAgents--;
            numAgents++;

            
            return m.pos;

        }
        


        return null;

    }

    public Point getCheckPoint(){

        return checkPoint;

    }

    public void move(Point position, Point where){
        for(MapAgent p : agents){
            if(p.pos.equals(p.pos, position)){
                p.pos.soma(where);
            }
        }
    }

    //Get the Order of the Matrix
    public int getOrder(){
        return dim;
    }

    //The Agent Can get the Map and see the obstacles
    public WorldMap[][] getMap(){
        return map;
    }


    private void initializeMap(String type){
        try {
            sc = new Scanner(new java.io.File(path+type));
        } catch (FileNotFoundException ex) {
            Logger.getLogger(Scenario.class.getName()).log(Level.SEVERE, null, ex);
        }
        
        dim = sc.nextInt();
        
        
        this.map = new WorldMap[dim][dim];
        

        for(int i=0; i<dim ; i++){
            for(int j=0; j<dim ; j++){
                int temp = sc.nextInt();
                if(temp == 0){
                    map[i][j] = new Spot(false);
                }
                else if(temp == 1){
                    map[i][j] = new Obstacle();
                }
                else if(temp == 2){
                    map[i][j] = new CheckPoint();
                }
                else if(temp == 3){
                    map[i][j] = new Spot(true);
                }

                
            }
            
        }
    }

   

    public synchronized boolean setMyNewPos(String name, Point oldPoint ,Point newPoint){

        //Putting The Barrier!
        //System.out.println("Agent "+name+" is trying to move from "+oldPoint+" to "+newPoint);
        if(numAgents != readyAgents.size()){
            return false;
        }


        if(map[newPoint.x][newPoint.y].getClass().toString().equals(MapAgent.class.toString())){
            return false;
        }
        else{
            map[oldPoint.x][oldPoint.y] = new Spot();
            map[newPoint.x][newPoint.y] = new MapAgent(name,newPoint);
            return true;
        }
        

        
    }

    public synchronized void killAgentPos(String name, Point p){
        map[p.x][p.y] = new Spot();
    }






}
